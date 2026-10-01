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
            (root / 'other_LevelTM_xray_C2S3_512').mkdir()
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                sequence.discover_pairs(root, 'LevelTM', 2, 10, 4)

    def test_auto_range_and_explicit_trial(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for session in (1, 2):
                for camera in (1, 2):
                    folder = root / f'test_KneeFlex_xray_02_C{camera}S{session}_512'
                    folder.mkdir()
                    for frame in (89, 90, 91):
                        (folder / f'image.{frame:04d}.tif').touch()
            self.assertEqual(len(sequence.discover_trials(root)), 2)
            trial = 'test_KneeFlex_xray_02_C1S2_512'
            pairs = sequence.discover_pairs(root, 'KneeFlex', trial=trial)
            self.assertEqual([p[0] for p in pairs], [89, 90, 91])
            self.assertIn('C2S2', str(pairs[0][2]))
            (pairs[-1][2]).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing requested'):
                sequence.discover_pairs(root, 'KneeFlex', trial=trial)

    def test_nested_root_and_duplicate_frame(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for camera in (1, 2):
                folder = root / 'SAM/tiffsUndistortedDownsampled' / f'test_LevelTM_xray_02_C{camera}S1_512'
                folder.mkdir(parents=True)
                (folder / 'image.0001.tif').touch()
            pairs = sequence.discover_pairs(root, 'LevelTM')
            self.assertEqual(len(pairs), 1)
            (pairs[0][1].parent / 'duplicate.0001.tiff').touch()
            with self.assertRaisesRegex(ValueError, 'Duplicate frame'):
                sequence.discover_pairs(root, 'LevelTM')

    def test_dry_run_external_output_and_no_model(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for camera in (1, 2):
                folder = root / f'test_LevelTM_xray_02_C{camera}S1_512'
                folder.mkdir()
                (folder / 'image.0007.tif').touch()
            output = root / 'results'
            args = ['--data-dir', str(root), '--activity', 'LevelTM', '--output', str(output), '--dry-run']
            sequence.main(args)
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                sequence.main(args)

    def test_interactive_selection(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for camera in (1, 2):
                folder = root / f'test_LevelTM_xray_02_C{camera}S1_512'
                folder.mkdir()
                (folder / 'image.0003.tif').touch()
            with patch('builtins.input', side_effect=[str(root), '1', str(root / 'exports')]):
                sequence.main(['--interactive', '--dry-run'])
            self.assertFalse((root / 'exports').exists())

    def test_invalid_ranges(self):
        for start, end, step in [(0, 10, 1), (10, 2, 1), (2, 10, 0)]:
            with self.assertRaises(ValueError):
                sequence.discover_pairs(ROOT, 'LevelTM', start, end, step)


if __name__ == '__main__':
    unittest.main()
