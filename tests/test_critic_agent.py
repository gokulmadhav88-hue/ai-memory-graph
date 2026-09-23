from src.reasoning.critic_agent import CriticAgent


def test_critic_agent_flags_retry_without_evidence():
    agent = CriticAgent()
    result = agent.assess("What happened?", evidence=[])

    assert result["sufficient"] is False
    assert result["retry"] is True
