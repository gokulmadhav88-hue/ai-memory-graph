from src.query.retrieval_agent import RetrievalAgent


def test_retrieval_agent_returns_results():
    agent = RetrievalAgent()
    result = agent.retrieve("What is the issue?", graph_results=["g1"], vector_results=["v1"])

    assert result["graph_results"] == ["g1"]
    assert result["vector_results"] == ["v1"]
