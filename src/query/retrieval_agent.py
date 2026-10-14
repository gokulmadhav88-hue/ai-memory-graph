"""Retrieval agent (Agent 4, Person B).

    retrieve(plan: RetrievalPlan, top_k: int = 5) -> EvidenceBundle

Reads graph facts (neo4j_store.query_subgraph) and/or vector chunks
(vector_store.query) according to plan["route"], merges them, filters noise,
and assigns unique evidence ids (E1.. for chunks, F1.. for graph facts).

* No Neo4j code and no LLM here. Stores are injectable for tests.
* No retry loop: the pipeline owns retries (it passes a new plan with attempt+1).
* Empty results are valid, never an error.
* Evidence text is DATA. It is only copied into the bundle, never interpreted.

Merge strategy is an open team decision; v1 is sequential: graph facts first,
then vector chunks (hybrid), with no rank fusion.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from src.stores import vector_store as _vector_store_module

logger = logging.getLogger(__name__)

VALID_ROUTES = ("graph", "vector", "hybrid")
DEFAULT_TOP_K = 5
GRAPH_DEPTH = 2
# Noise filtering knobs. MIN_CHUNK_SCORE depends on the embedding model (still an
# open decision), so treat 0.25 as a starting value to tune once a model is chosen.
MIN_CHUNK_SCORE = 0.25
MAX_GRAPH_FACTS = 20


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def _clean_plan(plan: Any) -> tuple[dict, list[str]]:
    """Return a safe copy of the plan plus notes about anything repaired."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a RetrievalPlan dict")
    notes: list[str] = []
    original = plan.get("original_question")
    original = original.strip() if isinstance(original, str) else ""
    rewritten = plan.get("rewritten_query")
    rewritten = rewritten.strip() if isinstance(rewritten, str) else ""
    query_text = rewritten or original
    route = plan.get("route")
    route = route.strip().lower() if isinstance(route, str) else ""
    if route not in VALID_ROUTES:
        notes.append(f"invalid route {plan.get('route')!r}, treated as hybrid")
        route = "hybrid"
    raw_entities = plan.get("target_entities")
    if isinstance(raw_entities, str):
        raw_entities = [raw_entities]
    entities: list[str] = []
    for e in raw_entities if isinstance(raw_entities, list) else []:
        if isinstance(e, str) and e.strip() and e.strip() not in entities:
            entities.append(e.strip())
    return {"query": query_text, "route": route, "entities": entities}, notes


# ---------------------------------------------------------------------------
# Graph side
# ---------------------------------------------------------------------------

def _default_graph_store() -> Any | None:
    """Real Neo4j store if Person A's module exposes query_subgraph; else None."""
    try:
        from src.stores import neo4j_store
    except Exception as e:                      # noqa: BLE001
        logger.warning("neo4j_store import failed: %s", e)
        return None
    if hasattr(neo4j_store, "query_subgraph"):
        return neo4j_store
    cls = getattr(neo4j_store, "Neo4jStore", None)
    if cls is None:
        return None
    try:
        instance = cls()
        connect = getattr(instance, "connect", None)
        if callable(connect):
            connect()
    except Exception as e:                      # noqa: BLE001
        logger.warning("Neo4jStore connection failed: %s", e)
        return None
    return instance if hasattr(instance, "query_subgraph") else None


def _fold(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum())


def _subgraph_to_facts(subgraph: Any, targets: list[str]) -> tuple[list[dict], bool]:
    """Subgraph -> (candidate facts ordered by hop distance, dropped_any).

    Noise filtering here: drop dangling edges, duplicates, and edges not reachable
    from a target entity within GRAPH_DEPTH hops; cap at MAX_GRAPH_FACTS (nearest first).
    If no target matches any node name/alias, hop filtering is skipped (nothing to anchor on).
    """
    if not isinstance(subgraph, dict):
        return [], False
    nodes = {n["node_id"]: n for n in subgraph.get("nodes") or []
             if isinstance(n, dict) and n.get("node_id") and n.get("name")}
    edges = [e for e in subgraph.get("edges") or [] if isinstance(e, dict)]
    dropped = False

    valid_edges = []
    for e in edges:
        if e.get("source_node_id") in nodes and e.get("target_node_id") in nodes and e.get("relation_type"):
            valid_edges.append(e)
        else:
            dropped = True

    # hop distance from target nodes (undirected BFS over the returned subgraph)
    target_keys = {_fold(t) for t in targets}
    start = [nid for nid, n in nodes.items()
             if _fold(n["name"]) in target_keys
             or any(_fold(a) in target_keys for a in n.get("aliases") or [] if isinstance(a, str))]
    adjacency: dict[str, set[str]] = {}
    for e in valid_edges:
        adjacency.setdefault(e["source_node_id"], set()).add(e["target_node_id"])
        adjacency.setdefault(e["target_node_id"], set()).add(e["source_node_id"])
    dist: dict[str, int] = {nid: 0 for nid in start}
    queue = deque(start)
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)

    scored: list[tuple[int, dict]] = []
    seen: dict[tuple, dict] = {}
    for e in valid_edges:
        s, t = e["source_node_id"], e["target_node_id"]
        if start:
            d = min(dist.get(s, 99), dist.get(t, 99))
            if d >= GRAPH_DEPTH:
                dropped = True
                continue
        else:
            d = 0
        subject, obj = nodes[s]["name"], nodes[t]["name"]
        relation = str(e["relation_type"]).strip().lower()
        key = (_fold(subject), relation, _fold(obj))
        note_ids = [n for n in e.get("source_note_ids") or [] if isinstance(n, str)]
        if key in seen:
            for n in note_ids:
                if n not in seen[key]["source_note_ids"]:
                    seen[key]["source_note_ids"].append(n)
            dropped = True                       # a duplicate was removed
            continue
        fact = {"subject": subject, "relation": relation, "object": obj, "source_note_ids": note_ids}
        seen[key] = fact
        scored.append((d, fact))

    scored.sort(key=lambda x: x[0])              # stable: nearest to the target first
    facts = [f for _, f in scored]
    if len(facts) > MAX_GRAPH_FACTS:
        facts = facts[:MAX_GRAPH_FACTS]
        dropped = True
    return facts, dropped


# ---------------------------------------------------------------------------
# Vector side
# ---------------------------------------------------------------------------

def _results_to_chunks(results: Any, min_score: float) -> tuple[list[dict], bool]:
    """Vector results -> (filtered chunks without ids, dropped_any)."""
    dropped = False
    best: dict[str, dict] = {}
    for r in results if isinstance(results, list) else []:
        if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not r["text"].strip() \
                or not r.get("note_id"):
            dropped = True
            continue
        try:
            score = max(0.0, min(1.0, float(r.get("score", 0.0))))
        except (TypeError, ValueError):
            dropped = True
            continue
        if score < min_score:
            dropped = True
            continue
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        chunk = {"note_id": str(r["note_id"]), "doc_id": str(meta.get("doc_id") or ""),
                 "section": meta.get("section") if isinstance(meta.get("section"), str) else None,
                 "text": r["text"], "score": score, "source": "vector"}
        prev = best.get(chunk["note_id"])
        if prev is None or score > prev["score"]:
            if prev is not None:
                dropped = True
            best[chunk["note_id"]] = chunk
        else:
            dropped = True
    chunks = sorted(best.values(), key=lambda c: c["score"], reverse=True)
    return chunks, dropped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(plan: dict, top_k: int = DEFAULT_TOP_K, *, vector_store: Any = None,
             graph_store: Any = None, min_score: float = MIN_CHUNK_SCORE) -> dict:
    """Plan -> EvidenceBundle. `vector_store` / `graph_store` / `min_score` are optional
    keyword-only overrides (tests); defaults are the real stores."""
    clean, notes = _clean_plan(plan)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        notes.append(f"invalid top_k {top_k!r}, used {DEFAULT_TOP_K}")
        top_k = DEFAULT_TOP_K
    route, query_text, targets = clean["route"], clean["query"], clean["entities"]

    if not query_text:
        return {"chunks": [], "graph_facts": [], "was_filtered": False, "route_used": route,
                "notes": "; ".join(notes + ["empty query, nothing to retrieve"])}

    want_graph = route in ("graph", "hybrid")
    want_vector = route in ("vector", "hybrid")
    facts: list[dict] = []
    chunks: list[dict] = []
    dropped_any = False
    graph_ok = vector_ok = False       # True = the source ran without error
    first_error: Exception | None = None

    # ---- graph ----
    if want_graph:
        if not targets:
            notes.append("no target entities, graph skipped")
        else:
            store = graph_store if graph_store is not None else _default_graph_store()
            if store is None:
                notes.append("graph store unavailable")
            else:
                try:
                    subgraph = store.query_subgraph(entity_names=targets, depth=GRAPH_DEPTH)
                    facts, d = _subgraph_to_facts(subgraph, targets)
                    dropped_any |= d
                    graph_ok = True
                except Exception as e:           # noqa: BLE001  (store boundary)
                    logger.warning("graph query failed: %s", e)
                    notes.append(f"graph query failed ({type(e).__name__})")
                    first_error = first_error or e

    # ---- vector (requested, or fallback when graph gave nothing) ----
    fallback = want_graph and not want_vector and not facts
    if want_vector or fallback:
        if route == "graph" and fallback:
            notes.append("graph returned nothing, used vector only")
        elif route == "hybrid" and not facts:
            notes.append("graph returned nothing, used vector only")
        store = vector_store if vector_store is not None else _vector_store_module.get_default_store()
        try:
            # Overfetch before status filtering so stale hits cannot crowd out current notes.
            status_store = graph_store
            check_active = getattr(status_store, "is_note_active", None)
            raw = store.query(query_text, top_k=top_k * 4 if callable(check_active) else top_k)
            if callable(check_active):
                active = [r for r in raw if isinstance(r, dict) and check_active(r.get("note_id"))]
                dropped_any |= len(active) != len(raw)
                raw = active
            chunks, d = _results_to_chunks(raw, min_score)
            dropped_any |= d
            vector_ok = True
        except Exception as e:                   # noqa: BLE001  (store boundary)
            logger.warning("vector query failed: %s", e)
            notes.append(f"vector query failed ({type(e).__name__})")
            first_error = first_error or e

    if not graph_ok and not vector_ok and first_error is not None:
        raise first_error                        # every attempted source failed: a real failure

    # Graph facts also carry their original source passages. This preserves dates,
    # qualifiers and quotes that are lost in a subject/predicate/object triple.
    get_note = getattr(graph_store, "get_note", None)
    if callable(get_note):
        seen_notes = {c["note_id"] for c in chunks}
        for fact in facts:
            for nid in fact["source_note_ids"]:
                note = get_note(nid)
                if nid in seen_notes or not note or not note.get("content"):
                    continue
                chunks.append({"note_id": nid, "doc_id": note["doc_id"], "section": note.get("section"),
                               "text": note["content"], "score": 1.0, "source": "graph"})
                seen_notes.add(nid)
    chunks = [{"evidence_id": f"E{i}", **c} for i, c in enumerate(chunks[:top_k], 1)]
    facts = [{"evidence_id": f"F{i}", **f} for i, f in enumerate(facts, 1)]

    if facts and any(c["source"] == "vector" for c in chunks):
        route_used = "hybrid"
    elif facts:
        route_used = "graph"
    elif chunks:
        route_used = "vector"
    else:
        route_used = route                       # nothing found: report what was asked for
    if not chunks and not facts:
        notes.append("no relevant evidence found")

    return {"chunks": chunks, "graph_facts": facts, "was_filtered": dropped_any,
            "route_used": route_used, "notes": "; ".join(notes) if notes else None}
