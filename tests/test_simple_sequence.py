import unittest
from unittest.mock import MagicMock

from iragent.agent import Agent
from iragent.message import Message
from iragent.models import SimpleSequentialAgents
from iragent.tools import get_time_now


class TestSimpleSequentialAgents(unittest.TestCase):
    def setUp(self):
        self.api_key = "fake-key"
        self.model = "fake-model"
        self.base_url = "https://api.fake.local/v1"

        # Mocking call_message to avoid actual OpenAI calls
        self.agent_a = Agent(
            name="time_reader",
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            system_prompt="Mock",
            fn=[get_time_now],
        )
        self.agent_a.call_message = MagicMock(
            return_value=Message(
                sender="time_reader",
                reciever="time_converter",
                content="The time is 12:00 PM",
                metadata={"reply_to": "0"},
            )
        )

        self.agent_b = Agent(
            name="time_converter",
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            system_prompt="Mock",
        )
        self.agent_b.call_message = MagicMock(
            return_value=Message(
                sender="time_converter",
                reciever="persian_translator",
                content="12:00",
                metadata={"reply_to": "0"},
            )
        )

        self.agent_c = Agent(
            name="persian_translator",
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            system_prompt="Mock",
        )
        self.agent_c.call_message = MagicMock(
            return_value=Message(
                sender="persian_translator",
                reciever="user",
                content="۱۲:۰۰",
                metadata={"reply_to": "0"},
            )
        )

    def test_agent_sequence(self):
        agents = [self.agent_a, self.agent_b, self.agent_c]
        manager = SimpleSequentialAgents(agents, init_message="what time is it?")
        result = manager.start()

        self.assertEqual(result.content, "۱۲:۰۰")
        self.assertEqual(result.sender, "persian_translator")
        self.assertEqual(result.reciever, "user")

        # Check that all agents were called
        self.agent_a.call_message.assert_called_once()
        self.agent_b.call_message.assert_called_once()
        self.agent_c.call_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
