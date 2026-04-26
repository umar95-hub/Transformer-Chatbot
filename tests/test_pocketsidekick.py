import json
import tempfile
import unittest
from pathlib import Path

from pocketsidekick import PocketSidekickApp


class PocketSidekickTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db_path = Path(self.temp_dir.name) / "memory.db"
        self.app = PocketSidekickApp(db_path=str(db_path))

    def test_router_and_tool_execution(self):
        response = self.app.handle_message("/tool:word_count one two three")
        self.assertIn("Tool 'word_count' => 3", response)

    def test_persona_mode_can_change(self):
        persona = self.app.set_persona(
            name="TravelBuddy",
            tone="warm and motivating",
            style_guide="Always include one actionable checklist item.",
        )
        self.assertEqual(persona["name"], "TravelBuddy")
        response = self.app.handle_message("hello")
        self.assertTrue(response.startswith("[TravelBuddy]"))

    def test_quantization_export(self):
        output = Path(self.temp_dir.name) / "quant" / "config.json"
        exported = self.app.export_quantization(str(output))
        self.assertEqual(exported, str(output))
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["method"], "int8")
        self.assertEqual(payload["bits"], 8)


if __name__ == "__main__":
    unittest.main()
