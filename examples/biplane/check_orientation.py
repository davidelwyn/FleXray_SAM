"""Compare a saved single-pass result with two quarter-turn predictions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import test_pair as pair


def restore_rotated(probability, shape, box, turns, pad_to_square):
    """Undo preprocessing in rotated coordinates before undoing the rotation."""
    x0, y0, x1, y1 = box
    crop_shape = (y1 - y0, x1 - x0)
    rotated_shape = crop_shape[::-1] if turns % 2 else crop_shape
    restored = pair.restore_probability(probability, rotated_shape, pad_to_square)
    result = np.zeros(shape, dtype='float32')
    result[y0:y1, x0:x1] = np.rot90(restored, -turns)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Existing frame folder with report.json')
    parser.add_argument('--view', choices=['view1', 'view2'], default='view1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direction', choices=['both', 'clockwise', 'anticlockwise'], default='both',
                        help='Limit the experiment to one additional prediction if desired')
    args = parser.parse_args()
    report = json.loads((args.baseline / 'report.json').read_text())
    if report['tta_samples'] != 1 or report['ensemble']:
        parser.error('Use a single-pass, single-model baseline for this small experiment.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output must be a new directory; existing paths are never overwritten.')
    info = report['views'][args.view]
    array = pair.load_image(info['input'])
    box = info.get('leg_crop', {}).get('box_xyxy', [0, 0, array.shape[1], array.shape[0]])
    x0, y0, x1, y1 = box
    model_input = pair.invert_image(array) if report['invert'] else array
    crop = model_input[y0:y1, x0:x1]
    field = (pair.load_image(args.baseline / args.view / 'detector_mask.png') > 0
             if report.get('mask_detector') else np.ones(array.shape, bool))
    labels = list(info['labels'])
    sys.path.insert(0, str(pair.ROOT / 'src'))
    from fxr.inference import FleXraySegmenter
    model = FleXraySegmenter.from_pretrained(report['model'], revision=report['revision'], device='cpu')
    if model.preprocessing.to_metadata() != report['preprocessing']:
        raise ValueError('Model preprocessing no longer matches the saved baseline.')
    output.mkdir(parents=True)
    variants = [('baseline', 0)]
    if args.direction in ('both', 'anticlockwise'):
        variants.append(('plus90', 1))
    if args.direction in ('both', 'clockwise'):
        variants.append(('minus90', -1))
    sheet = Image.new('RGB', (512 * len(labels), 544 * len(variants)), 'black')
    stats = {'baseline': str(args.baseline.resolve()), 'view': args.view,
             'crop_box_xyxy': box, 'new_predictions': len(variants) - 1, 'variants': {},
             'note': 'Positive angle is counterclockwise; outputs are in original coordinates. No ground-truth accuracy measured.'}
    for row, (name, turns) in enumerate(variants):
        folder = output / name
        folder.mkdir()
        probabilities = None
        if turns:
            rotated = np.rot90(crop, turns).copy()
            pair.display_image(rotated).save(folder / 'model_input_preview.png')
            probabilities = model.predict(rotated, threshold=report['threshold'], tta_samples=1,
                                          seed=report['seed']).probabilities.detach().cpu().numpy()[0]
        stats['variants'][name] = {}
        for column, label in enumerate(labels):
            if not turns:
                mask = pair.load_image(args.baseline / args.view / f'{label}_mask.png') > 0
            else:
                probability = restore_rotated(probabilities[model.label_names.index(label)],
                                              array.shape, box, turns, model.preprocessing.pad_to_square)
                np.save(folder / f'{label}_probability.npy', probability)
                mask = probability >= report['threshold']
                Image.fromarray(mask.astype('uint8') * 255).save(folder / f'{label}_raw_mask.png')
                mask &= field
                if report.get('keep_largest'):
                    mask = pair.largest_component(mask)
                if report.get('fill_holes'):
                    mask = pair.fill_mask_holes(mask) & field
            Image.fromarray(mask.astype('uint8') * 255).save(folder / f'{label}_mask.png')
            pixels = np.array(pair.display_image(array))
            color = {'femurs': (0, 100, 255), 'tibiae': (255, 0, 0)}.get(label, (0, 255, 90))
            pixels[mask] = (pixels[mask] * 0.55 + np.array(color) * 0.45).astype('uint8')
            overlay = Image.fromarray(pixels)
            overlay.save(folder / f'{label}_overlay.png')
            overlay.thumbnail((512, 512))
            sheet.paste(overlay, (column * 512, row * 544 + 32))
            ImageDraw.Draw(sheet).text((column * 512 + 8, row * 544 + 8), f'{name}: {label}', fill='white')
            stats['variants'][name][label] = {'mask_pixels': int(mask.sum())}
        print(f'Finished {name}', flush=True)
    sheet.save(output / 'comparison.png')
    (output / 'report.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
