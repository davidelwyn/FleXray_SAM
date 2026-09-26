"""Run independent biplane segmentations over a matched frame range."""
from __future__ import annotations

import copy
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw

import test_pair


def discover_pairs(data_dir, activity, start, end, step):
    """Resolve camera files by frame ID; reject ambiguity and missing frames."""
    if start < 1 or end < start or step < 1:
        raise ValueError('Require 1 <= start <= end and step >= 1.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', activity):
        raise ValueError('Activity must be a simple name, e.g. LevelTM or KneeFlex.')
    cameras = []
    for camera in (1, 2):
        folders = sorted(data_dir.glob(f'*_{activity}_*C{camera}S*'))
        folders = [folder for folder in folders if folder.is_dir()]
        if len(folders) != 1:
            raise ValueError(f'Expected one {activity} camera {camera} folder; found {len(folders)}.')
        frames = {}
        for path in folders[0].iterdir():
            match = re.search(r'\.(\d+)\.tiff?$', path.name, flags=re.IGNORECASE)
            if path.is_file() and match:
                frame = int(match.group(1))
                if frame in frames:
                    raise ValueError(f'Duplicate frame {frame} in {folders[0]}.')
                frames[frame] = path.resolve()
        cameras.append(frames)
    selected = list(range(start, end + 1, step))
    missing = {f'camera{i + 1}': [n for n in selected if n not in frames]
               for i, frames in enumerate(cameras)}
    if any(missing.values()):
        raise ValueError(f'Missing requested frames: {missing}')
    return [(n, cameras[0][n], cameras[1][n]) for n in selected]


def main(argv=None):
    """Load the model once, export every pair, and create a browsable review."""
    parser = test_pair.parser()
    parser.description = __doc__
    parser.add_argument('--activity', required=True, help='Activity name in folder names, e.g. LevelTM')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True, help='Inclusive last frame')
    parser.add_argument('--step', type=int, default=1)
    parser.set_defaults(output=test_pair.ROOT / 'outputs/biplane' /
                        ('sequence_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    args = parser.parse_args(argv)
    if args.view1 or args.view2 or args.ground_truth or args.prepare_only:
        parser.error('Sequence mode uses --activity and frame range; use test_pair.py for explicit paths, reference masks or prepare-only.')
    pairs = discover_pairs(args.data_dir, args.activity, args.start, args.end, args.step)
    output = args.output.resolve()
    if not output.is_relative_to(test_pair.ROOT) or output == test_pair.ROOT:
        raise ValueError('Output must be a new directory inside FleXray_SAM.')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(test_pair.ROOT / 'src'))
    from fxr.inference import FleXraySegmenter
    segmenter = FleXraySegmenter.from_pretrained(args.model, revision=args.revision,
                                               device=args.device, ensemble=args.ensemble)
    output.mkdir(parents=True)
    rows, cards, pages = [], [], []
    previous = {}
    sheet = None
    for index, (frame, first, second) in enumerate(pairs):
        pair_args = copy.copy(args)
        pair_args.view1, pair_args.view2 = str(first), str(second)
        pair_args.output = output / f'frame_{frame:04d}'
        report = test_pair.run(pair_args, segmenter)
        for view, info in report['views'].items():
            for label, stats in info['labels'].items():
                key = (view, label)
                area = stats['mask_pixels']
                prior = previous.get(key)
                flags = []
                if not area:
                    flags.append('empty_mask')
                if info.get('leg_crop', {}).get('status', '').startswith('fallback'):
                    flags.append('crop_fallback')
                if info.get('detector_mask', {}).get('status', '').startswith('fallback'):
                    flags.append('detector_fallback')
                if prior is not None and max(area, prior) > 2 * max(1, min(area, prior)):
                    flags.append('area_change_over_2x')
                previous[key] = area
                rows.append({'frame': frame, 'view': view, 'label': label,
                             'mask_pixels': area, 'search_area_reduction': stats['search_area_reduction'],
                             'removed_island_pixels': stats['removed_island_pixels'],
                             'removed_detector_pixels': stats['removed_detector_pixels'],
                             'review_flags': ';'.join(flags)})
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if index % 8 == 0:
            sheet = Image.new('RGB', (1024, 8 * 296), 'black')
        with Image.open(pair_args.output / 'comparison.png') as image:
            image.thumbnail((1024, 272))
            sheet.paste(image, (0, (index % 8) * 296 + 24))
        ImageDraw.Draw(sheet).text((8, (index % 8) * 296 + 5), f'Frame {frame}', fill='white')
        if index % 8 == 7 or index == len(pairs) - 1:
            name = f'review_{index // 8 + 1:02d}.jpg'
            used = (index % 8 + 1) * 296
            sheet.crop((0, 0, 1024, used)).save(output / name, quality=95)
            pages.append(name)
        cards.append(f'<h2>Frame {frame}</h2><a href="frame_{frame:04d}/comparison.png">'
                     f'<img loading="lazy" src="frame_{frame:04d}/comparison.png" width="100%"></a>')
        print(f'Completed {index + 1}/{len(pairs)} pairs', flush=True)
    (output / 'review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Biplane sequence review</title>'
        '<body style="background:#171717;color:white;font-family:Arial;margin:24px">'
        f'<h1>{html.escape(args.activity)}: frames {args.start}-{args.end}</h1>'
        '<p>Columns: view1 femur / tibia, view2 femur / tibia for default labels. '
        'Green: predicted bone. Yellow: search box. Independent predictions, not tracking. '
        'Flags in summary.csv are review prompts, not accuracy scores.</p>'
        + ''.join(cards) + '</body>', encoding='utf-8')
    (output / 'sequence.json').write_text(json.dumps({
        'activity': args.activity, 'frames': [n for n, _, _ in pairs],
        'review_pages': pages, 'flags_count': sum(bool(row['review_flags']) for row in rows),
        'note': 'No ground truth: segmentation and pose accuracy are not measured.'}, indent=2), encoding='utf-8')
    print(f'Review: {output / "review.html"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f'Error: {exc}')
