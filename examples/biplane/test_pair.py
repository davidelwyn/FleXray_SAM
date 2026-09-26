"""Inspect one synchronized biplane pair with FleXray; see README.md."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path.home() / 'OneDrive - Cardiff University/PDRA/EPSRC-KneeImp/FleXray_testdata'


def load_image(path):
    """Read a single grayscale image without reducing its bit depth or rotating it."""
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f'{path}: supply a single frame, not a stack.')
        array = np.array(image)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f'{path}: expected a finite 2D grayscale image.')
    return array


def display_image(array):
    """Window a copy for visualization only."""
    low, high = np.percentile(array, [0.5, 99.5])
    pixels = np.clip((array.astype(float) - low) / max(high - low, 1e-8), 0, 1)
    return Image.fromarray((pixels * 255).astype('uint8')).convert('RGB')


def restore_probability(probability, shape, pad_to_square):
    """Invert model resizing, then remove the exact original integer padding."""
    height, width = shape
    side = max(shape)
    canvas = (side, side) if pad_to_square else (width, height)
    restored = np.asarray(Image.fromarray(np.asarray(probability, dtype='float32')).resize(
        canvas, Image.Resampling.BILINEAR))
    top, left = ((side - height) // 2, (side - width) // 2) if pad_to_square else (0, 0)
    return restored[top:top + height, left:left + width].copy()


def invert_image(array):
    """Reverse intensity without changing coordinates or mutating source pixels."""
    if array.dtype.kind == 'u':
        return np.iinfo(array.dtype).max - array
    values = array.astype('float64')
    return (values.min() + (values.max() - values)).astype(array.dtype)


def search_region(mask, margin):
    """Use all detected components; an empty detection falls back to the full image."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        box = [0, 0, width, height]
    else:
        box = [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(width, int(xs.max()) + 1 + margin), min(height, int(ys.max()) + 1 + margin)]
    roi = np.zeros_like(mask, dtype=bool)
    x0, y0, x1, y1 = box
    roi[y0:y1, x0:x1] = True
    return box, roi


def largest_component(mask):
    """Keep the largest 8-connected foreground island; preserve empty masks.

    Equal-size islands are resolved in row-major scan order.
    """
    from scipy import ndimage as ndi

    components, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax()


def score(mask, roi, truth):
    """Measure segmentation agreement and how much annotated bone the ROI retains."""
    intersection = int((mask & truth).sum())
    total = int(mask.sum() + truth.sum())
    union = int((mask | truth).sum())
    return {'dice': 2 * intersection / total if total else 1.0,
            'iou': intersection / union if union else 1.0,
            'roi_bone_recall': float((roi & truth).sum() / truth.sum()) if truth.any() else None}


def brightness_crop(array, cutoff=0.65, margin=20):
    """Find a dark leg inside a near-black-bordered detector; fail open if uncertain.

    Thresholds refer to the original polarity, scaled by its observed range.
    This heuristic assumes one dominant leg and bright background, not any X-ray.
    """
    from scipy import ndimage as ndi

    def largest(mask):
        labels, count = ndi.label(mask)
        if not count:
            return np.zeros_like(mask)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == sizes.argmax()

    height, width = array.shape
    full = [0, 0, width, height]
    values = array.astype('float64')
    span = float(values.max() - values.min())
    if span == 0:
        return np.ones(array.shape, bool), full, {'status': 'fallback_constant_image'}
    scaled = (values - values.min()) / span
    detector = ndi.binary_fill_holes(largest(scaled > 0.03))
    # Remove the dark detector rim before finding the leg, so it cannot join
    # both sides of the image into one artificial foreground component.
    rim_pixels = max(2, round(min(array.shape) * 0.06))
    inner = ndi.distance_transform_edt(np.pad(detector, 1))[1:-1, 1:-1] > rim_pixels
    dark = inner & (ndi.median_filter(scaled, size=5) < cutoff)
    leg = ndi.binary_fill_holes(largest(dark))
    fraction = float(leg.sum() / max(int(inner.sum()), 1))
    metadata = {'status': 'cropped', 'brightness_cutoff': cutoff,
                'detector_cutoff': 0.03, 'rim_exclusion_pixels': rim_pixels,
                'leg_fraction_of_inner_detector': fraction}
    if detector.mean() < 0.2 or fraction < 0.05 or fraction > 0.9:
        metadata['status'] = 'fallback_unreliable_leg_mask'
        return np.ones(array.shape, bool), full, metadata
    expanded = ndi.binary_dilation(leg, iterations=margin) if margin else leg
    box, _ = search_region(expanded, 0)
    return expanded, box, metadata


def detector_mask(array, inset=3):
    """Estimate the nonblack detector field in original polarity, filling dark anatomy.

    Follow the measured field boundary rather than assuming a centered circle.
    Unreliable estimates fail open to avoid silently discarding anatomy.
    """
    from scipy import ndimage as ndi

    values = array.astype('float64')
    span = float(values.max() - values.min())
    metadata = {'status': 'applied', 'cutoff': 0.03, 'inset_pixels': inset}
    if span == 0:
        metadata['status'] = 'fallback_constant_image'
        return np.ones(array.shape, bool), metadata
    field = ndi.binary_fill_holes(largest_component((values - values.min()) / span > 0.03))
    if field.mean() < 0.2:
        metadata['status'] = 'fallback_small_detector'
        return np.ones(array.shape, bool), metadata
    if inset:
        field = ndi.distance_transform_edt(np.pad(field, 1))[1:-1, 1:-1] > inset
    if field.mean() < 0.2:
        metadata['status'] = 'fallback_excessive_inset'
        return np.ones(array.shape, bool), metadata
    metadata['retained_fraction'] = float(field.mean())
    return field, metadata


def restore_crop_probability(probability, shape, crop_box, pad_to_square):
    """Undo model padding/resizing and paste a crop into native detector coordinates."""
    x0, y0, x1, y1 = crop_box
    result = np.zeros(shape, dtype='float32')
    result[y0:y1, x0:x1] = restore_probability(
        probability, (y1 - y0, x1 - x0), pad_to_square)
    return result


def resolve_pair(args):
    """Require explicit paths or an unambiguous matching frame in C1S1/C2S1 folders."""
    if bool(args.view1) != bool(args.view2):
        raise ValueError('Supply both --view1 and --view2.')
    if args.view1:
        paths = [Path(p) if Path(p).is_absolute() else args.data_dir / p
                 for p in (args.view1, args.view2)]
    else:
        paths = []
        for camera in ('C1S1', 'C2S1'):
            matches = sorted(args.data_dir.glob(f'**/*{camera}*.{args.frame:04d}.tif'))
            if len(matches) != 1:
                raise ValueError(f'Expected one {camera} frame {args.frame}, found {len(matches)}. '
                                 'Use --view1 and --view2 to choose the pair explicitly.')
            paths.append(matches[0])
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError('The two camera inputs must be different files.')
    return paths


def run(args, segmenter=None):
    """Run both views independently and save native-coordinate masks and a report."""
    if not 0 < args.threshold < 1 or args.margin < 0 or args.tta_samples < 1:
        raise ValueError('Require 0 < threshold < 1, margin >= 0 and tta-samples >= 1.')
    if not 0.03 < args.leg_cutoff < 1 or args.crop_margin < 0:
        raise ValueError('Require 0.03 < leg-cutoff < 1 and crop-margin >= 0.')
    if args.detector_inset < 0:
        raise ValueError('detector-inset must be nonnegative.')
    if len(set(args.labels)) != len(args.labels) or any(
            not re.fullmatch(r'[A-Za-z0-9_-]+', label) for label in args.labels):
        raise ValueError('Labels must be unique simple names (letters, numbers, underscore, hyphen).')
    paths = resolve_pair(args)
    arrays = [load_image(path) for path in paths]
    truths = {}
    if args.ground_truth:
        for view, array in zip(('view1', 'view2'), arrays):
            for label in args.labels:
                truth = load_image(args.ground_truth / view / f'{label}.png')
                if truth.shape != array.shape:
                    raise ValueError(f'Ground truth for {view}/{label} must match input dimensions.')
                truths[view, label] = truth > 0
    if not args.prepare_only and segmenter is None:
        # Use this checkout, even if another FleXray version is installed.
        sys.path.insert(0, str(ROOT / 'src'))
        try:
            from fxr.inference import FleXraySegmenter
        except ImportError as exc:
            raise RuntimeError('Install dependencies first: python -m pip install -e .') from exc
        segmenter = FleXraySegmenter.from_pretrained(
            args.model, revision=args.revision, device=args.device, ensemble=args.ensemble)
    if not args.prepare_only:
        missing = set(args.labels) - set(segmenter.label_names or ())
        if missing:
            raise ValueError(f'Unknown labels: {sorted(missing)}. Available: {segmenter.label_names}')
    output = args.output.resolve()
    # Keep test outputs in this checkout, never in the source-image folder.
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Output must be a new directory inside the FleXray_SAM repository.')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'inputs_only' if args.prepare_only else 'predicted',
              'model': args.model, 'revision': args.revision, 'ensemble': args.ensemble,
              'threshold': args.threshold, 'margin_pixels': args.margin,
              'tta_samples': args.tta_samples, 'seed': args.seed,
              'invert': args.invert,
              'leg_crop': args.leg_crop,
              'keep_largest': args.keep_largest,
              'mask_detector': args.mask_detector,
              'coordinates': 'original pixels; origin top-left; boxes [x0,y0,x1,y1], end exclusive',
              'note': 'Independent 2D segmentations; no calibrated 3D pose or SAM import is performed.',
              'views': {}}
    if not args.prepare_only:
        report['preprocessing'] = segmenter.preprocessing.to_metadata()
        report['device'] = str(segmenter.device)
    panels = []
    for view, path, array in zip(('view1', 'view2'), paths, arrays):
        folder = output / view
        folder.mkdir()
        preview = display_image(array)
        preview.save(folder / 'input_preview.png')
        model_input = invert_image(array) if args.invert else array
        info = {'input': str(path.resolve()), 'shape': list(array.shape), 'dtype': str(array.dtype)}
        field = np.ones(array.shape, bool)
        if args.mask_detector:
            field, info['detector_mask'] = detector_mask(array, args.detector_inset)
            Image.fromarray(field.astype('uint8') * 255).save(folder / 'detector_mask.png')
            field_preview = np.array(preview)
            field_preview[~field] = (field_preview[~field] * 0.4 + np.array([255, 0, 0]) * 0.6).astype('uint8')
            Image.fromarray(field_preview).save(folder / 'detector_preview.png')
        crop_box = [0, 0, array.shape[1], array.shape[0]]
        if args.leg_crop:
            leg, crop_box, crop_info = brightness_crop(array, args.leg_cutoff, args.crop_margin)
            crop_info.update({'box_xyxy': crop_box, 'margin_pixels': args.crop_margin})
            info['leg_crop'] = crop_info
            Image.fromarray(leg.astype('uint8') * 255).save(folder / 'leg_mask.png')
            leg_overlay = np.array(preview)
            leg_overlay[leg] = (leg_overlay[leg] * 0.7 + np.array([0, 150, 255]) * 0.3).astype('uint8')
            leg_overlay = Image.fromarray(leg_overlay)
            cx0, cy0, cx1, cy1 = crop_box
            ImageDraw.Draw(leg_overlay).rectangle((cx0, cy0, cx1 - 1, cy1 - 1), outline='yellow', width=2)
            leg_overlay.save(folder / 'leg_crop_preview.png')
            model_input = model_input[cy0:cy1, cx0:cx1]
        display_image(model_input).save(folder / 'model_input_preview.png')
        report['views'][view] = info
        if not args.prepare_only:
            start = time.perf_counter()
            prediction = segmenter.predict(model_input, threshold=args.threshold,
                                           tta_samples=args.tta_samples, seed=args.seed)
            probabilities = prediction.probabilities.detach().cpu().numpy()[0]
            info['inference_seconds'] = time.perf_counter() - start
            info['labels'] = {}
            for label in args.labels:
                probability = restore_crop_probability(probabilities[segmenter.label_names.index(label)],
                                                       array.shape, crop_box, segmenter.preprocessing.pad_to_square)
                mask = probability >= args.threshold
                raw_pixels = int(mask.sum())
                if args.keep_largest or args.mask_detector:
                    Image.fromarray(mask.astype('uint8') * 255).save(folder / f'{label}_raw_mask.png')
                mask &= field
                detector_removed = raw_pixels - int(mask.sum())
                before_islands = int(mask.sum())
                if args.keep_largest:
                    mask = largest_component(mask)
                box, roi = search_region(mask, args.margin)
                roi &= field
                np.save(folder / f'{label}_probability.npy', probability)
                Image.fromarray(mask.astype('uint8') * 255).save(folder / f'{label}_mask.png')
                Image.fromarray(roi.astype('uint8') * 255).save(folder / f'{label}_roi.png')
                # Preserve native dimensions and intensity values for later integration experiments.
                Image.fromarray(np.where(roi, array, 0)).save(folder / f'{label}_roi_image.tif')
                overlay = np.array(preview)
                overlay[mask] = (overlay[mask] * 0.55 + np.array([0, 255, 90]) * 0.45).astype('uint8')
                overlay = Image.fromarray(overlay)
                x0, y0, x1, y1 = box
                ImageDraw.Draw(overlay).rectangle((x0, y0, x1 - 1, y1 - 1), outline='yellow', width=2)
                overlay.save(folder / f'{label}_overlay.png')
                stats = {'box_xyxy': box, 'mask_pixels': int(mask.sum()),
                         'raw_mask_pixels': raw_pixels,
                         'removed_detector_pixels': detector_removed,
                         'removed_island_pixels': before_islands - int(mask.sum()),
                         'search_area_reduction': float(1 - roi.mean()),
                         'status': 'detected' if mask.any() else 'empty_prediction_full_frame_fallback'}
                if (view, label) in truths:
                    stats.update(score(mask, roi, truths[view, label]))
                info['labels'][label] = stats
                panels.append((f'{view}: {label}', overlay))
        else:
            panels.append((f'{view}: input only', preview))
    sheet = Image.new('RGB', (512 * len(panels), 544), 'black')
    for index, (title, panel) in enumerate(panels):
        panel.thumbnail((512, 512))
        sheet.paste(panel, (index * 512, 32))
        ImageDraw.Draw(sheet).text((index * 512 + 8, 8), title, fill='white')
    sheet.save(output / 'comparison.png')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {report["status"]} results to {output}')
    return report


def parser():
    """Build the single-pair experiment CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--view1', help='Absolute path or path relative to --data-dir')
    p.add_argument('--view2', help='Absolute path or path relative to --data-dir')
    p.add_argument('--frame', type=int, default=1, help='Matching numbered TIFF frame (default: 1)')
    p.add_argument('--labels', nargs='+', default=['femurs', 'tibiae'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--margin', type=int, default=20, help='Bounding-box margin in original pixels')
    p.add_argument('--model', default='VictorButoi/flexray', help='Hugging Face ID or local bundle')
    p.add_argument('--revision')
    p.add_argument('--ensemble', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--tta-samples', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--invert', action='store_true', help='Invert intensities before preprocessing; exports retain original intensities')
    p.add_argument('--keep-largest', action='store_true', help='Keep only the largest 8-connected island per bone and view before computing search boxes')
    p.add_argument('--mask-detector', action='store_true', help='Exclude pixels outside the measured detector field before island cleanup; also clip ROI masks')
    p.add_argument('--detector-inset', type=int, default=3, help='Inset detector boundary by this many original pixels (default: 3)')
    p.add_argument('--leg-crop', action='store_true', help='Crop using the dark leg against bright background, before inversion')
    p.add_argument('--leg-cutoff', type=float, default=0.65, help='Normalized original-intensity cutoff for leg mask (default: 0.65)')
    p.add_argument('--crop-margin', type=int, default=20, help='Expand leg mask before cropping, in native pixels')
    p.add_argument('--ground-truth', type=Path, help='Folder containing view1/LABEL.png and view2/LABEL.png')
    p.add_argument('--prepare-only', action='store_true', help='Check real input loading without model inference')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/biplane' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return p


if __name__ == '__main__':
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, OSError, RuntimeError) as exc:
        cli.exit(1, f'Error: {exc}\n')
