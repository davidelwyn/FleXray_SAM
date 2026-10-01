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
import imageio.v2 as imageio
import numpy as np

import test_pair


def discover_trials(data_dir):
    """List paired camera folders without silently choosing repeats or sessions.

    Prefer direct children. Otherwise check the conventional SAM image root.
    Selecting noHDR explicitly avoids mixing processed copies with originals.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    for root in (data_dir, data_dir / 'tiffsUndistortedDownsampled',
                 data_dir / 'SAM' / 'tiffsUndistortedDownsampled'):
        trials = []
        if not root.is_dir():
            continue
        for first in sorted(root.iterdir()):
            match = re.fullmatch(r'(.+?)_([^_]+)_xray_(.*?)C1(S\d+.*)', first.name)
            if not first.is_dir() or not match:
                continue
            second = first.with_name(first.name.replace('_C1S', '_C2S'))
            if second.is_dir():
                trials.append({'activity': match[2], 'name': first.name,
                               'cameras': (first, second)})
        if trials:
            return trials
    raise ValueError(f'No paired *_ACTIVITY_xray_*_C1S*/C2S* folders found under {data_dir}.')


def choose_item(items, prompt, label=str):
    for index, item in enumerate(items, 1):
        print(f'  {index}. {label(item)}')
    while True:
        answer = input(prompt + ' (number): ').strip()
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        print('Choose one of the listed numbers.')


def discover_pairs(data_dir, activity, start=None, end=None, step=1, trial=None):
    """Resolve camera files by frame ID; reject ambiguity and missing frames."""
    if (start is not None and start < 1) or (end is not None and end < 1) or step < 1 or (
            start is not None and end is not None and end < start):
        raise ValueError('Require 1 <= start <= end and step >= 1.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', activity):
        raise ValueError('Activity must be a simple name, e.g. LevelTM or KneeFlex.')
    cameras = []
    trials = [t for t in discover_trials(data_dir) if t['activity'] == activity
              and (trial is None or t['name'] == trial)]
    if len(trials) != 1:
        raise ValueError(f'Expected one paired trial for {activity}; found {len(trials)}. '
                         'Use --interactive or --trial with the C1 folder name.')
    for camera in (1, 2):
        folder = trials[0]['cameras'][camera - 1]
        frames = {}
        for path in folder.iterdir():
            match = re.search(r'\.(\d+)\.tiff?$', path.name, flags=re.IGNORECASE)
            if path.is_file() and match:
                frame = int(match.group(1))
                if frame in frames:
                    raise ValueError(f'Duplicate frame {frame} in {folder}.')
                frames[frame] = path.resolve()
        cameras.append(frames)
    if not all(cameras):
        raise ValueError('Both camera folders must contain numbered TIFF frames.')
    start = min(min(c) for c in cameras) if start is None else start
    end = max(max(c) for c in cameras) if end is None else end
    if end < start:
        raise ValueError('End frame precedes start frame.')
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
    parser.add_argument('--interactive', action='store_true', help='Prompt for root, activity, trial and output folder')
    parser.add_argument('--activity', help='Activity name in folder names, e.g. LevelTM')
    parser.add_argument('--trial', help='Exact C1 folder name when an activity has multiple trials')
    parser.add_argument('--start', type=int, help='First frame (default: discovered minimum)')
    parser.add_argument('--end', type=int, help='Last frame (default: discovered maximum)')
    parser.add_argument('--list-activities', action='store_true', help='List available trials without inference')
    parser.add_argument('--dry-run', action='store_true', help='Validate selection and show frame count without inference')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--video-fps', type=float, default=10,
                        help='Frame rate for the side-by-side biplane overlay video (default: 10)')
    parser.set_defaults(output=None, data_dir=None)
    args = parser.parse_args(argv)
    interactive = args.interactive or (argv is None and len(sys.argv) == 1)
    if interactive:
        if args.data_dir is None:
            root_answer = input('Root folder containing camera folders: ').strip().strip('"')
            if not root_answer:
                parser.error('Choose a data root folder.')
            args.data_dir = Path(root_answer).expanduser()
        trials = discover_trials(args.data_dir)
        if args.activity is None:
            args.activity = choose_item(sorted({t['activity'] for t in trials}), 'Select activity')
        candidates = [t for t in trials if t['activity'] == args.activity]
        if not candidates:
            parser.error('No paired trials for that activity.')
        if args.trial is None:
            selected = candidates[0] if len(candidates) == 1 else choose_item(
                candidates, 'Select trial/session', lambda t: t['name'])
            args.trial = selected['name']
        if args.output is None and not args.list_activities:
            destination = input('Output parent folder (a new run subfolder will be created): ').strip().strip('"')
            if not destination:
                parser.error('Choose an output folder.')
            args.output = Path(destination).expanduser() / ('sequence_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if args.data_dir is None:
        parser.error('Provide --data-dir ROOT or use --interactive.')
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.list_activities:
        for trial in discover_trials(args.data_dir):
            print(f'{trial["activity"]}: {trial["name"]}')
        return
    if not args.activity:
        parser.error('Provide --activity or use --interactive.')
    if args.video_fps <= 0:
        parser.error('--video-fps must be greater than zero.')
    if args.view1 or args.view2 or args.ground_truth or args.prepare_only:
        parser.error('Sequence mode uses --activity and frame range; use test_pair.py for explicit paths, reference masks or prepare-only.')
    pairs = discover_pairs(args.data_dir, args.activity, args.start, args.end, args.step, args.trial)
    args.start, args.end = pairs[0][0], pairs[-1][0]
    print(f'{args.activity}: {len(pairs)} frame pairs, {args.start} to {args.end}, step {args.step}.')
    if args.output is None:
        parser.error('Provide --output NEW_FOLDER or use --interactive.')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    print(f'Output: {output}')
    if args.dry_run:
        return
    sys.path.insert(0, str(test_pair.ROOT / 'src'))
    from fxr.inference import FleXraySegmenter
    segmenter = FleXraySegmenter.from_pretrained(args.model, revision=args.revision,
                                               device=args.device, ensemble=args.ensemble)
    args.orientation_direction = {
        'view1': args.view1_rotation,
        'view2': args.view2_rotation,
    }
    output.mkdir(parents=True)
    rows, cards, pages = [], [], []
    previous = {}
    sheet = None
    video_path = output / 'biplane_overlay_video.mp4'
    with imageio.get_writer(video_path, fps=args.video_fps, codec='libx264',
                            quality=8, macro_block_size=None) as video:
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
                                 'filled_hole_pixels': stats['filled_hole_pixels'],
                                 'review_flags': ';'.join(flags)})
            with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            # One clean video frame: C1 and C2, each with all bone masks overlaid.
            video_panels = []
            for view in ('view1', 'view2'):
                with Image.open(pair_args.output / view / 'combined_overlay.png') as panel:
                    panel = panel.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
                    video_panels.append(panel.copy())
            video_frame = Image.new('RGB', (1024, 544), 'black')
            drawer = ImageDraw.Draw(video_frame)
            for column, (name, panel) in enumerate(zip(('C1', 'C2'), video_panels)):
                video_frame.paste(panel, (column * 512, 32))
                drawer.text((column * 512 + 8, 8), name, fill='white')
            drawer.text((468, 8), f'Frame {frame}', fill='white')
            video.append_data(np.asarray(video_frame))
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
        '<video controls preload="metadata" style="width:100%;max-width:1024px" '
        'src="biplane_overlay_video.mp4"></video>'
        '<p>Columns: view1 femur / tibia, view2 femur / tibia for default labels. '
        'Blue: femur. Red: tibia. Green: other labels. Yellow: search box. Independent predictions, not tracking. '
        'Flags in summary.csv are review prompts, not accuracy scores.</p>'
        + ''.join(cards) + '</body>', encoding='utf-8')
    (output / 'sequence.json').write_text(json.dumps({
        'activity': args.activity, 'frames': [n for n, _, _ in pairs],
        'data_root': str(args.data_dir), 'camera_folders': [str(p.parent) for p in pairs[0][1:]],
        'review_pages': pages, 'biplane_overlay_video': video_path.name,
        'video_fps': args.video_fps,
        'flags_count': sum(bool(row['review_flags']) for row in rows),
        'note': 'No ground truth: segmentation and pose accuracy are not measured.'}, indent=2), encoding='utf-8')
    print(f'Review: {output / "review.html"}', flush=True)
    print(f'Biplane overlay video: {video_path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        sys.exit(f'Error: {exc}')
    except KeyboardInterrupt:
        sys.exit('Cancelled.')
