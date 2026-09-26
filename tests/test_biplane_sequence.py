"""Frame pairing checks that do not download or run a model."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_biplane_example import example, ROOT

spec = importlib.util.spec_from_file_location('biplane_sequence', ROOT / 'examples/biplane/run_sequence.py')
sequence = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'test_pair': example}):
    spec.loader.exec_module(sequence)


class SequenceTest(unittest.TestCase):
    def test_numeric_pairing_and_missing_frames(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as work:
            root = Path(work)
            for camera in (1, 2):
                folder = root / f'test_LevelTM_xray_C{camera}S2_512'
                folder.mkdir()
                for frame in (10, 2, 6):
                    (folder / f'image.{frame:04d}.tif').touch()
            pairs = sequence.discover_pairs(root, 'LevelTM', 2, 10, 4)
            self.assertEqual([p[0] for p in pairs], [2, 6, 10])
            self.assertIn('C2S2', str(pairs[0][2]))
            with self.assertRaisesRegex(ValueError, 'Missing requested'):
                sequence.discover_pairs(root, 'LevelTM', 2, 10, 1)
            (root / 'other_LevelTM_xray_C1S3_512').mkdir()
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                sequence.discover_pairs(root, 'LevelTM', 2, 10, 4)

    def test_invalid_ranges(self):
        for start, end, step in [(0, 10, 1), (10, 2, 1), (2, 10, 0)]:
            with self.assertRaises(ValueError):
                sequence.discover_pairs(ROOT, 'LevelTM', start, end, step)


if __name__ == '__main__':
    unittest.main()
