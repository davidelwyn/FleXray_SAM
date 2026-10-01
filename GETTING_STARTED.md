# FleXray to SAM: shared workflow

This repository adds an experimental biplane knee workflow to the original FleXray project. Start here; the original library documentation and attribution remain in README.md.

## 1. Install

From the repository root, use Python 3.10 or newer in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

See [Windows setup](examples/biplane/VSCODE_SETUP.md). The first inference run downloads the pretrained model unless `--model` points to an existing local bundle.

## 2. Select a root, activity and output folder

```powershell
.\.venv\Scripts\python.exe examples/biplane/run_sequence.py --interactive
```

Enter the data root, choose an activity from the numbered menu, choose a trial/session if more than one exists, and enter an output parent folder. The program creates a unique run subfolder and discovers the first frame, last frame and number of paired frames automatically. Running with no arguments also opens this menu. These are terminal prompts, so no desktop GUI package is needed.

Expected camera folders:

```text
data-root/
  SUBJECT_KneeFlex_xray_02_C1S1_512/
    image.0001.tif
    image.0002.tif
  SUBJECT_KneeFlex_xray_02_C2S1_512/
    image.0001.tif
    image.0002.tif
```

Direct children are preferred. If none are found, discovery checks `tiffsUndistortedDownsampled/` and `SAM/tiffsUndistortedDownsampled/`. To use an alternative copy such as `noHDR`, select that folder explicitly. Matching frame numbers are assumed to represent synchronized cameras; verify this for your acquisition.

For repeatable runs without prompts:

```powershell
python examples/biplane/run_sequence.py --data-dir "D:\Xrays" --activity KneeFlex --output "D:\Results\knee_test_01"
```

The CLI output path must not already exist. `--list-activities` lists available trial names. Use `--trial EXACT_C1_FOLDER_NAME` when needed. `--start 20 --end 60` selects a subset; otherwise the range is discovered. `--step 5` samples every fifth frame. Missing requested frames, duplicate frame IDs and ambiguous trials stop the run before model loading. Frames are never silently dropped to make the two cameras agree.

Add `--dry-run` to validate selection and print the count without creating outputs or downloading/running a model.

Preprocessing remains explicit. The earlier experimental combination can be selected with:

```powershell
python examples/biplane/run_sequence.py --interactive --invert --leg-crop --keep-largest --mask-detector --fill-holes --auto-orientation --view1-rotation clockwise --view2-rotation counterclockwise --device cpu
```

These settings were explored for particular acquisitions; they are not validated defaults for every activity. See [the detailed biplane guide](examples/biplane/README.md) for individual controls and limitations.

## 3. Review the predictions

Open `review.html` or `biplane_overlay_video.mp4` in the selected output folder. Review `summary.csv` for empty masks, crop fallbacks and abrupt area changes. Each `frame_####` contains the original-coordinate femur/tibia masks, probabilities, overlays and a report. Inspect both cameras before using the masks for SAM.

## 4. Export the masked trial in MATLAB

Run `masked_trial_for_sam` from the repository root. The implementation and editable settings are in [examples/matlab/masked_trial_for_sam.m](examples/matlab/masked_trial_for_sam.m).

Choose the data root (or SAM folder), the activity/trial configuration, the matching Python sequence output, and an output parent folder. The available mask frame count is discovered automatically. Set `dilationPixels` at the top; the current value is 30 pixels. Use `frameNumbers = []` for all available mask frames.

A unique export folder contains:

- `femur/`: femur-only masked TIFFs, binary masks, copied SAM configuration and frame status.
- `tibia/`: tibia-only equivalents.
- `combined/`: union of both masks and equivalent exports.

Images retain original size, orientation and intensity inside the mask. An empty prediction becomes a background-only frame and is recorded in `frame_status.csv`, preserving camera alignment. Missing files or mismatched dimensions still stop the export. The copied configurations retain both original bone-model entries; the variants change the X-ray images, not which models SAM loads.

Open the generated `.cfg` in SAM yourself. The script does not launch Slicer. Absolute resource paths refer to the local calibration/volumes/models; generated configurations are not portable to another machine without those resources and path updates. MATLAB requires Image Processing Toolbox. Exports have not yet been verified end to end in SAM.

Blanking pixels is a visual masking experiment. It does not establish that SAM's similarity calculation excludes those pixels. True tracking exclusion and any effect of the artificial mask boundary require separate testing.

## Repository layout and sharing

- `src/fxr/`: original FleXray library.
- `examples/biplane/`: Python segmentation, sequence processing and orientation experiments.
- `examples/matlab/`: masked SAM export and optional comparison GIF/MP4 generation.
- `tests/test_biplane_*.py`: offline checks using synthetic inputs/fake predictions.
- `docs/PROJECT_NOTES.md`: OneNote-ready progress summary and next steps.
- Root MATLAB scripts: compatibility launchers.

Share the tracked source with Git or a source archive, not the entire working folder. Local environments and generated outputs are ignored. Keep acquisition data, subject models, calibration and results outside the repository. Reports contain local source paths. Preserve LICENSE and CITATION.cff and the original FleXray attribution. No commit, push or publication is performed by this cleanup.
