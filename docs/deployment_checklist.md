# Deployment checklist

Target: a private shared team workspace first, followed by a separate public-user
release. APP_ENV supports local or team. Public user data isolation is not implemented.

## Before the first team deployment

1. **Review/version the changes.** Teammates should review contracts, supported
   formats, identity/update rules and agent_capabilities.md. Commit a branch and
   merge a reviewed pull request after checks pass. Keep .env and
   .streamlit/secrets.toml out of Git and container builds.
2. **Configure real services.** Set private model keys and Neo4j local/Aura
   credentials on a staging instance. Use neo4j+s for Aura and one
   CHROMA_COLLECTION per workspace/embedding schema. Grant the schema/write
   permissions required by the adapter.
3. **Run actual database/model checks.** Use the commands below. The live adapter
   test creates/deletes only its own random namespace. Model probing makes a small
   paid request to each distinct provider/model. Passing means compatibility,
   not answer-quality acceptance.
4. **Prepare embeddings.** Run warmup before user traffic and persist its cache.
   Default: local all-MiniLM-L6-v2. Evaluate it on your languages/documents;
   changing embeddings needs a new collection and reingestion.
5. **Configure team OIDC and authorization.** Set APP_ENV=team,
   APP_AUTH_REQUIRED=1, APP_ALLOWED_EMAILS to a comma-separated team allowlist.
   Copy .streamlit/secrets.example.toml to secrets.toml; configure your provider,
   HTTPS redirect ending /oauth2callback, strong cookie secret, client ID/secret,
   and metadata URL. The app requires email_verified=true.
6. **Test real-browser access.** Signed-out visitors cannot access uploads/chat;
   verified allowlisted accounts can; outsiders/unverified accounts cannot. Test
   logout and removal from the allowlist. All allowed members share documents
   and replacement permissions.
7. **Evaluate representative ground truth.** Extend eval/demo_cases.jsonl with
   actual expected terms, document IDs and abstention cases. Include multi-hop,
   conflicts, dated updates, duplicate names, noisy/empty inputs, tables/scans.
   Manually check claims against citations. Keyword checks are not semantic accuracy.
8. **Measure real load/cost.** Establish acceptance targets for retrieval recall,
   supported answers, abstention, median/p95 latency, tokens/question, and ingestion
   throughput. UI timings/tokens count logical model calls; SDK retries can add
   network attempts. There is no monetary cost calculation or hard spending quota.
9. **Back up and restore.** Persist Neo4j, Chroma and model cache. Test restoration
   on staging: note/source counts, citations, memory status, and pending index work.
   Pause writes for coordinated volume backups; avoid copying live Chroma files.
10. **Use HTTPS/private ports.** Put the loopback-bound app behind a TLS reverse
    proxy supporting WebSockets. Keep Neo4j Bolt/Browser ports private. Supply
    private secret files or hosting secrets. Verify non-root container volume permissions.
11. **Build/test/restart on the chosen host.** Docker is unavailable in this audited
    workspace: image and Compose files are prepared but not built here. Simulate
    database/index/model outages; restart and retry pending indexing. Verify
    replacement keeps the previous version after failure. Decide source/audit
    retention and privacy rules for data sent to model providers.
12. **Add operational controls.** Monitor errors, partial ingestion,
    vector_pending/vector_cleanup_pending, revision conflicts, resources, latency
    and token budgets. Set upstream rate limits and provider spending limits.
    Document ownership, pausing uploads, alert response, and rollback.

## Windows commands from the inner project directory

These need private service configuration. Live Neo4j is opt-in.

    .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    .\.venv\Scripts\python.exe -m pip check
    .\.venv\Scripts\python.exe -m src.cli warmup
    .\.venv\Scripts\python.exe -m src.cli check --probe-model
    $env:RUN_NEO4J_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest tests/test_neo4j_adapter.py -q
    .\.venv\Scripts\python.exe -m src.cli ingest data/raw_documents/demo_atlas.txt
    .\.venv\Scripts\python.exe -m src.cli ask "Who manages Atlas?"
    .\.venv\Scripts\python.exe -m eval.run_eval eval/demo_cases.jsonl

For local Neo4j plus the app container:

    docker compose -f compose.yaml -f compose.team.yaml config
    docker compose -f compose.yaml -f compose.team.yaml up --build -d

The overlay uses neo4j://neo4j:7687 and persists Chroma/cache. It requires .env
and real .streamlit/secrets.toml. Inspect rendered configuration locally: it can
include secrets. For Aura, run the app image separately with the Aura URI and
equivalent secret/volume mounts, omitting the local DB/dependency.

HTTP health checks verify the server, not full database/model readiness. Run CLI
and document-to-answer smoke checks after deploying.

## Before a public-user release

- Define accounts, workspaces and roles. Derive corpus access from authenticated
  server-side identity; users cannot choose another user's graph/collection.
  Add cross-account leakage tests for retrieval, citations, upload, replacement,
  deletion and export.
- Enforce document-level permissions when members have different access.
  OIDC identifies users; authorization remains the application's responsibility.
- Replace full-corpus snapshots/in-memory traversal with incremental database
  operations before larger corpus or multiple-worker scale. Establish safe hosted
  vector storage and ingestion/job coordination.
- Add background jobs/status/cancellation, bounded workers, user quotas,
  request rate limits, abuse protection and spending controls.
- Isolate document parsing workers and test malicious/decompression-heavy files;
  byte-size/archive limits alone are not complete resource or malware protection.
- Add document management and data deletion/export with defined retention.
  Decide whether persistent conversation memory belongs in the product.
- Implement observability, alerts, incident/recovery runbooks, migrations,
  upgrade/rollback testing, hosted load testing and security assessment.
- Gate release on real annotated quality tests and adversarial inputs. Decide
  supported OCR/vision/web options; do not advertise unconfigured callbacks.

## Current limits

Small shared team corpus, one app process. Projection/index updates batch, but
the full graph snapshot is still serialized and kept in Python. Graph integration
decisions serialize; retrieval is bounded approximate search. Live model quality,
Neo4j, OIDC, Docker, Linux CI and hosted performance remain unverified.

Setup references: [OpenAI SDK](https://developers.openai.com/api/reference/python)
and [Streamlit OIDC](https://docs.streamlit.io/develop/concepts/connections/authentication).
