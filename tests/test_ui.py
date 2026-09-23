from pathlib import Path
import tempfile
import unittest

from saturnia_ui import NO_LORA, _safe_dimensions, discover_loras


class UiHelpersTests(unittest.TestCase):
    def test_discovers_checkpoints_in_a_local_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / ".runs" / "output" / "sample_run"
            run.mkdir(parents=True)
            (run / "sample_run_250.safetensors").touch()
            (run / "sample_run.safetensors").touch()
            choices = discover_loras(Path(directory))
            self.assertEqual(len(choices), 2)
            self.assertTrue(any("step 250" in choice.label for choice in choices))
            self.assertTrue(any("final" in choice.label for choice in choices))

    def test_empty_checkout_has_no_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(discover_loras(Path(directory)), [])

    def test_dimensions_are_multiple_of_sixteen(self):
        self.assertEqual(_safe_dimensions(1025, 769), (1024, 768))

    def test_base_model_label(self):
        self.assertIn("base model", NO_LORA)


if __name__ == "__main__":
    unittest.main()
