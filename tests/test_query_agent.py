from src.query.query_agent import QueryAgent


def test_query_agent_routes_hybrid():
    agent = QueryAgent()
    route = agent.route("What are the key entities?")

    assert route == "hybrid"
