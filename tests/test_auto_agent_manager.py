import unittest
from unittest.mock import MagicMock, patch

from iragent.agent import Agent
from iragent.message import Message
from iragent.models import AutoAgentManager
from iragent.tools import get_time_now, simple_termination


class TestAutoAgentManager(unittest.TestCase):
    def setUp(self):
        self.agent_a = Agent(
            name="time_reader",
            model="gpt-4.1-mini",
            base_url="mock-url",
            api_key="mock-key",
            system_prompt="Mock agent A",
            fn=[get_time_now],
        )
        self.agent_b = Agent(
            name="time_converter",
            model="gpt-4.1-mini",
            base_url="mock-url",
            api_key="mock-key",
            system_prompt="Mock agent B",
        )
        self.agent_c = Agent(
            name="persian_translator",
            model="gpt-4.1-mini",
            base_url="mock-url",
            api_key="mock-key",
            system_prompt="Mock agent C",
        )

        # Mock the call_message function to return fake messages
        self.agent_a.call_message = MagicMock(
            return_value=Message(
                sender="time_reader",
                reciever="time_converter",
                content="2024-07-15 13:00:00",
                metadata={"message_id": "1"},
            )
        )

        self.agent_b.call_message = MagicMock(
            return_value=Message(
                sender="time_converter",
                reciever="persian_translator",
                content="15 Tir 1403",
                metadata={"message_id": "2"},
            )
        )

        self.agent_c.call_message = MagicMock(
            return_value=Message(
                sender="persian_translator",
                reciever="user",
                content="15 تیر 1403 [#finish#]",
                metadata={"message_id": "3"},
            )
        )

    @patch("iragent.models.Agent.call_message")
    def test_auto_agent_routing(self, mock_auto_router):
        # Mock the auto router responses
        mock_auto_router.side_effect = [
            Message(
                sender="router", reciever=None, content="time_converter", metadata={}
            ),
            Message(
                sender="router",
                reciever=None,
                content="persian_translator",
                metadata={},
            ),
            Message(sender="router", reciever=None, content="user", metadata={}),
        ]

        manager = AutoAgentManager(
            init_message="What time is it now?",
            agents=[self.agent_a, self.agent_b, self.agent_c],
            first_agent=self.agent_a,
            max_round=5,
            termination_fn=simple_termination,
            termination_word="[#finish#]",
        )

        result = manager.start()
        self.assertIn("[#finish#]", result.content)
