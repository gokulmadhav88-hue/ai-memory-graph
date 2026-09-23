from src.ingestion.graph_agent import GraphAgent


def test_graph_agent_write_returns_summary():
    agent = GraphAgent()
    result = agent.write(["note_1", "note_2"])

    assert result["written"] == 2
