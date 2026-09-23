from pathlib import Path
import tempfile
import unittest

from saturnia_lora.dataset import content_only_caption, image_size


class DatasetTests(unittest.TestCase):
    def test_content_only_caption_removes_explicit_style_sentence(self):
        raw = (
            "SATURNIA_STYLE, A moth rests on a leaf. "
            "Fine graphite hatching and coral washes cover warm paper."
        )
        self.assertEqual(
            content_only_caption(raw, "SATURNIA_STYLE"),
            "SATURNIA_STYLE. A moth rests on a leaf.",
        )

    def test_png_dimensions_without_pillow(self):
        # PNG signature + IHDR length/type + width/height is enough for parser.
        payload = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\x0dIHDR"
            b"\x00\x00\x04\x00\x00\x00\x03\x00"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            path.write_bytes(payload)
            self.assertEqual(image_size(path), (1024, 768))


if __name__ == "__main__":
    unittest.main()
