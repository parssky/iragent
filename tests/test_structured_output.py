from unittest.mock import MagicMock, patch

from iragent.agent import Agent, AgentFactory
from iragent.message import Message


def mock_create_agent(*args, **kwargs):
    assert kwargs.get("response_format") == {"type": "json_object"}, "Missing or incorrect response_format"

    mock_agent = MagicMock(spec=Agent)
    mock_agent.call_message.return_value = Message(content={"fruits": ["apple", "banana"]})
    return mock_agent


@patch("iragent.agent.AgentFactory.create_agent", side_effect=mock_create_agent)
def test_agent_supports_response_format(mock_create):
    base_url = "mock-url"
    api_key = "mock-key"
    model = "mock-model"
    provider = "AvalAI"
    prompt = "You extract fruits and return JSON."

    # Instantiate factory and create the agent
    factory = AgentFactory(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
    )

    agent = factory.create_agent(
        name="fruit_extractor",
        system_prompt=prompt,
        response_format={"type": "json_object"},
    )

    # Test call_message behavior
    message = Message(content="I have an apple and a banana.")
    response = agent.call_message(message)

    # Final assertion
    assert isinstance(response.content, dict)
    assert response.content["fruits"] == ["apple", "banana"]