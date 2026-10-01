"""Offline tests of native-coordinate exports, independent of model downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('biplane_example', ROOT / 'examples/biplane/test_pair.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class BiplaneTest(unittest.TestCase):
    def test_fill_enclosed_holes_but_not_open_notches(self):
        mask = np.zeros((10, 10), bool)
        mask[2:8, 2:8] = True
        mask[4, 4] = False
        mask[2:4, 6] = False
        result = example.fill_mask_holes(mask)
        self.assertTrue(result[4, 4])
        self.assertFalse(result[3, 6])
        self.assertFalse(result[0, 0])
        self.assertFalse(mask[4, 4])

    def test_detector_mask_keeps_dark_anatomy_and_rejects_exterior(self):
        yy, xx = np.mgrid[:100, :100]
        circle = (xx - 50) ** 2 + (yy - 50) ** 2 <= 40 ** 2
        source = np.zeros((100, 100), 'uint8')
        source[circle] = 220
        source[40:60, 40:60] = 0
        field, metadata = example.detector_mask(source, inset=3)
        self.assertEqual(metadata['status'], 'applied')
        self.assertTrue(field[50, 50])
        self.assertFalse(field[0, 0])
        self.assertFalse(field[50, 90])
        self.assertTrue(field[50, 85])
        blank, metadata = example.detector_mask(np.zeros((10, 10), 'uint8'))
        self.assertTrue(blank.all())
        self.assertIn('fallback', metadata['status'])

    def test_largest_island_and_search_box(self):
        source = np.zeros((20, 20), bool)
        source[5:10, 6:11] = True
        source[18, 18] = True
        cleaned = example.largest_component(source)
        self.assertEqual(int(cleaned.sum()), 25)
        self.assertEqual(int(source.sum()), 26)
        self.assertEqual(example.search_region(cleaned, 1)[0], [5, 4, 12, 11])
        self.assertFalse(example.largest_component(np.zeros((4, 4), bool)).any())
        # Diagonal neighbors belong to one island with 8-connectivity.
        np.testing.assert_array_equal(example.largest_component(np.eye(4, dtype=bool)),
                                      np.eye(4, dtype=bool))
        # Equal areas: retain the first component in scan order.
        tied = np.zeros((4, 4), bool)
        tied[0, 0] = tied[3, 3] = True
        self.assertTrue(example.largest_component(tied)[0, 0])
        self.assertEqual(int(example.largest_component(tied).sum()), 1)

    def test_brightness_mask_excludes_black_border_and_bright_background(self):
        yy, xx = np.mgrid[:120, :120]
        detector = (xx - 60) ** 2 + (yy - 60) ** 2 < 55 ** 2
        source = np.zeros((120, 120), 'uint8')
        source[detector] = 240
        source[detector & (xx >= 45) & (xx <= 75)] = 60
        mask, box, metadata = example.brightness_crop(source, margin=5)
        self.assertEqual(metadata['status'], 'cropped')
        self.assertTrue(mask[60, 60])
        self.assertFalse(mask[60, 15])
        self.assertFalse(mask[0, 0])
        self.assertLessEqual(box[0], 45)
        self.assertGreaterEqual(box[2], 76)
        self.assertLess(box[2] - box[0], 60)
        _, box, metadata = example.brightness_crop(np.zeros((20, 30), 'uint8'))
        self.assertEqual(box, [0, 0, 30, 20])
        self.assertEqual(metadata['status'], 'fallback_constant_image')

    def test_crop_predictions_restore_native_offsets(self):
        restored = example.restore_crop_probability(np.ones((4, 4), 'float32'),
                                                    (10, 12), [3, 2, 7, 6], True)
        expected = np.zeros((10, 12), 'float32')
        expected[2:6, 3:7] = 1
        np.testing.assert_array_equal(restored, expected)

    def test_inversion_preserves_source_and_depth(self):
        for dtype, maximum in [('uint8', 255), ('uint16', 65535)]:
            source = np.array([[0, 10, maximum]], dtype=dtype)
            original = source.copy()
            inverted = example.invert_image(source)
            np.testing.assert_array_equal(inverted, [[maximum, maximum - 10, 0]])
            np.testing.assert_array_equal(source, original)
            np.testing.assert_array_equal(example.invert_image(inverted), source)
            self.assertEqual(inverted.dtype, source.dtype)

    def test_padding_and_boxes(self):
        # Odd padding: top=1 and bottom=2 for a 4x7 image on a 7x7 canvas.
        probability = np.zeros((7, 7), dtype='float32')
        probability[1:5, 2:4] = 1
        restored = example.restore_probability(probability, (4, 7), True)
        np.testing.assert_array_equal(restored, probability[1:5])
        box, roi = example.search_region(restored > 0.5, 1)
        self.assertEqual(box, [1, 0, 5, 4])
        self.assertEqual(int(roi.sum()), 16)
        box, roi = example.search_region(np.zeros((4, 7), bool), 1)
        self.assertEqual(box, [0, 0, 7, 4])
        self.assertTrue(roi.all())

    def test_unpadded_resize_and_scores(self):
        restored = example.restore_probability(np.ones((2, 2), 'float32'), (5, 9), False)
        np.testing.assert_array_equal(restored, np.ones((5, 9)))
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        scores = example.score(mask, np.ones_like(mask), truth)
        self.assertAlmostEqual(scores['dice'], 2 / 3)
        self.assertEqual(scores['iou'], 0.5)
        self.assertEqual(scores['roi_bone_recall'], 1)

    def test_pair_exports(self):
        class FakeTensor:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self):
                array = np.zeros((1, 2, 7, 7), 'float32')
                array[0, 0, 1:5, 2:4] = 1
                return array

        class FakeSegmenter:
            label_names = ('femurs', 'tibiae')
            device = 'cpu'
            preprocessing = SimpleNamespace(pad_to_square=True, to_metadata=lambda: {'pad_to_square': True})
            def predict(self, image, **kwargs):
                assert image.dtype == np.uint16
                return SimpleNamespace(probabilities=FakeTensor())

        # A collaborator may export outside the checkout; still refuse overwrite.
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            original = np.arange(28, dtype='uint16').reshape(4, 7) * 1000
            for camera in ('C1S1', 'C2S1'):
                Image.fromarray(original).save(work / f'example_{camera}.0001.tif')
            args = example.parser().parse_args(['--data-dir', str(work), '--output', str(work / 'result'), '--margin', '0'])
            report = example.run(args, FakeSegmenter())
            self.assertEqual(report['status'], 'predicted')
            self.assertEqual(report['views']['view1']['labels']['femurs']['box_xyxy'], [2, 0, 4, 4])
            self.assertEqual(report['views']['view2']['labels']['tibiae']['search_area_reduction'], 0)
            exported = example.load_image(work / 'result/view1/femurs_roi_image.tif')
            self.assertEqual(exported.dtype, original.dtype)
            np.testing.assert_array_equal(exported[:, 2:4], original[:, 2:4])
            self.assertTrue((exported[:, :2] == 0).all())
            self.assertEqual(json.loads((work / 'result/report.json').read_text())['status'], 'predicted')
            with self.assertRaises(FileExistsError):
                example.run(args, FakeSegmenter())

    def test_reject_stack_and_ambiguous_pair(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as work:
            work = Path(work)
            image = Image.fromarray(np.zeros((4, 4), 'uint8'))
            path = work / 'stack.tif'
            image.save(path, save_all=True, append_images=[image])
            with self.assertRaises(ValueError):
                example.load_image(path)
            args = example.parser().parse_args(['--data-dir', str(work)])
            with self.assertRaises(ValueError):
                example.resolve_pair(args)


if __name__ == '__main__':
    unittest.main()
