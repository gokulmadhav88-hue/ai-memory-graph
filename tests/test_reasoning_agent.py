from src.reasoning.reasoning_agent import ReasoningAgent


def test_reasoning_agent_answers_with_evidence():
    agent = ReasoningAgent()
    answer = agent.answer("What happened?", evidence=[{"source": "doc_1"}])

    assert "evidence" in answer.lower()
    assert "1" in answer
