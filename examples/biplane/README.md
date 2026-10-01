# Biplane segmentation and SAM preview

Start with [the shared workflow guide](../../GETTING_STARTED.md) for root/activity/output selection and MATLAB export. See [Windows setup](VSCODE_SETUP.md) for installation.

For a single pair, run from the repository root:

```powershell
python examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 1
```

Select an unambiguous pair with --view1 and --view2 when needed; paths are absolute or relative to --data-dir. The default data directory is the current working directory.

Inputs must be single-frame grayscale images readable by Pillow (e.g. TIFF,
PNG, BMP, JPEG). TIFF stacks, color images, and DICOM are not supported by this
small test script. Image intensities and orientation are passed unchanged to
FleXray's bundle-defined preprocessing. Only visualization copies are contrast
windowed. The model is loaded once; each view is segmented independently.
The first inference run downloads pretrained weights from Hugging Face. Use
`--model "C:\path\to\bundle"` for an existing local portable model bundle,
`--device cpu` to force CPU, or `--ensemble --tta-samples 8` for a slower test.
`--revision` can pin a model commit and `--seed` controls augmentation randomness.

## Inspecting the result

Each run creates a new folder under `outputs/biplane` in this repository.
`--output` can select a new directory anywhere writable; existing directories
are rejected so earlier experiments cannot be overwritten. The input files are
never modified.

- `comparison.png`: views and selected bones side by side; blue femur, red tibia
  (green for other labels),
  yellow padded rectangular search region.
- `view1/` and `view2/`: each contains a display preview and one set of outputs
  per bone: floating-point probability `.npy`, binary `*_mask.png`, binary
  `*_roi.png`, native-intensity `*_roi_image.tif`, and `*_overlay.png`.
- `report.json`: input paths, dimensions, model settings, preprocessing,
  inference timings, original-pixel bounding boxes, detected area, and search
  area reduction. Reduction is a fraction: 0.6 means 60% fewer image pixels in
  the rectangular search region. It is not a measured speedup or pose accuracy.

Masks and probabilities are restored to original image dimensions before
thresholding and region calculation, undoing the model's square padding.
Boxes use `[x0, y0, x1, y1]` with exclusive upper bounds and a top-left origin.
All components are retained; plural labels may include more than one bone of
the same class. The masks are independent and may overlap. No left/right
instance identity is inferred. Empty detections are explicitly flagged and
retain the full image as a fallback. Inspect false positives and missed edges
before choosing a margin or threshold.

## Measuring quality

### Test a sequence

Run from the repository root (one line):

```powershell
.\.venv\Scripts\python.exe examples/biplane/run_sequence.py --data-dir "D:\Xrays" --output "D:\Results\new_run" --activity LevelTM --start 20 --end 60 --invert --leg-crop --keep-largest --tta-samples 1 --device cpu
```

Omit `--start` and `--end` to discover all frames, or set them to choose an inclusive range. `--step 5` samples
every fifth frame. `--activity KneeFlex` selects the other activity. There must
be exactly one folder per camera for the activity; missing/duplicate frames are
rejected before inference. The model loads once for the whole run.

Open `review.html` in the output folder to scroll through all results, or inspect
the numbered `review_*.jpg` contact sheets. Each `frame_####` folder contains the
usual masks and report. `summary.csv` lists each camera/bone result and flags
empty masks, crop fallbacks, and greater-than-twofold area changes between sampled
frames. These are review prompts, not validated failure detection or accuracy
scores. Unflagged masks can still be wrong. This runs independent segmentation,
not temporal propagation, and frame-number matching assumes synchronized inputs.

Each sequence run also creates `biplane_overlay_video.mp4`. It shows C1 and C2
side by side with the femur in blue and tibia in red on the same image, without
search boxes. Set its playback rate with `--video-fps 10`.

For frame-by-frame leg orientation correction, add `--auto-orientation`. The
undirected leg-mask long axis is estimated independently in each view and frame.
If it is within 45 degrees of horizontal (angles are normalized so -160 degrees
is treated as 20 degrees from horizontal), that crop is rotated clockwise by 90 degrees
before inference; otherwise it is left as-is. The probability map is rotated
back to original detector coordinates before thresholds, cleanup, and export.
`report.json` records the estimated angle, selected turn, and policy for every
view/frame. The option makes one model prediction per view/frame. This direction
is based on the promising KneeFlex C1 checks at frames 173, 175 and 177; it has
not been established for every camera or activity. Review the angle and output
around orientation changes before using these masks for tracking. Change the
horizontal threshold with `--orientation-horizontal-deg 40` if needed.
Rotation direction is set independently with `--view1-rotation` and
`--view2-rotation`; each accepts `clockwise` or `counterclockwise`. For the
current KneeFlex hypothesis, use clockwise for C1 and counterclockwise for C2.

Example for the requested range:

```powershell
.\.venv\Scripts\python.exe examples/biplane/run_sequence.py --data-dir "D:\Xrays" --output "D:\Results\new_run" --activity KneeFlex --start 120 --end 180 --invert --leg-crop --keep-largest --mask-detector --fill-holes --auto-orientation --view1-rotation clockwise --view2-rotation counterclockwise --tta-samples 1 --device cpu
```

### Exclude the detector exterior

Add `--mask-detector` to either the pair or sequence command. It estimates the
visible field from original intensities, retains the largest nonblack region,
and fills interior holes so dark anatomy is not excluded. It follows the actual
field boundary rather than assuming an exactly centered circle. A default
3-pixel inward margin removes the immediate rim; adjust `--detector-inset`, or
use 0 to retain the measured boundary. Inspect `detector_preview.png` (red is
excluded) and `detector_mask.png` before relying on the result. This assumes a
near-black exterior; constant or implausibly small fields fall back to no filtering.

Filtering is applied before largest-island selection and also clips exported
ROI masks/images. Boxes are still rectangular and may cross the circle; the
ROI masks themselves exclude the exterior. Empty bone predictions retain the
full-detector ROI. Saved probabilities remain unfiltered, and raw masks are
preserved. The report distinguishes detector-removed and island-removed pixels.
This cannot remove incorrect pixels that lie inside the detector field.

### Fill enclosed holes

Add `--fill-holes` to fill background regions completely enclosed by each bone
mask, after largest-island filtering. The result is clipped to the detector
mask again. This does not bridge open notches, recover missing shafts, or join
separate islands. It fills all enclosed holes, so inspect whether a gap represents
a true feature before using it. Raw masks and probability maps remain available;
the report records `filled_hole_pixels` separately from removed pixels.

For the combined sequence experiment:

```powershell
.\.venv\Scripts\python.exe examples/biplane/run_sequence.py --data-dir "D:\Xrays" --output "D:\Results\new_run" --activity LevelTM --start 20 --end 60 --invert --leg-crop --keep-largest --mask-detector --fill-holes --tta-samples 1 --device cpu
```

### Keep the largest island

Add `--keep-largest` to retain the largest connected region for each bone in
each camera separately. Diagonal neighbors count as connected (8-connectivity).
Filtering happens after native-size probability thresholding and before search
boxes, overlays, ROI exports and reference-mask scores are calculated.
`*_raw_mask.png` preserves the unfiltered thresholded mask; `*_mask.png` is the
cleaned mask. Probability arrays remain unchanged. The report records removed
pixel counts and the option used. Empty predictions still fall back to a full-frame
search region; equal-size components use the first in scan order.

This is useful for isolated background detections. It will not remove an artifact
connected to the main mask, fill holes, or guarantee that the largest region is
the correct bone. It can discard valid disconnected anatomy, especially if both
legs are visible, so it is off by default.

### Bright-background leg crop

Add `--leg-crop --invert --tta-samples 1` to test a brightness-guided crop.
The heuristic finds the nonblack detector, fills its interior, removes its dark
rim, and selects the largest darker component inside it. It expands that leg
mask by `--crop-margin 20` pixels and runs inference on the bounding rectangle.
No pixels inside the crop are hard-masked. The original image files are unchanged.
This assumes a dark leg against bright background in the ORIGINAL image,
regardless of `--invert`. It is not a universal foreground detector.

Inspect `view1/leg_crop_preview.png` and `view2/leg_crop_preview.png`: blue is
the expanded leg mask and yellow is the inference crop. `leg_mask.png` saves
the expanded mask, and `model_input_preview.png` shows the actual cropped input
polarity. The report records the crop coordinates and fallback status.
Unreliable or constant-image detections fall back to the full image. Plausible
but incorrect masks are still possible: visually check that the crop retains
the target anatomy. Increase `--leg-cutoff` from its default 0.65 to retain
brighter tissue, or increase `--crop-margin` for a more generous region.

Probabilities and bone masks are restored to original coordinates, with zero
probability outside the crop. Exported ROI TIFFs retain original dimensions
and intensities. The leg mask does not itself identify bone or suppress pixels
inside the crop. A leg spanning the full detector height may still produce a
tall crop and little gain in model resolution. Validate against manual masks
before using the resulting search regions for tracking.

To compare intensity polarity, add `--invert` to the same image-pair command.
To average eight augmented predictions, add `--tta-samples 8 --seed 0`.
Test each change separately before combining them. Inversion is applied before
the bundle preprocessing (unsigned images use dtype maximum minus pixel value;
other images reflect around their observed range). The report records `invert`.
`model_input_preview.png` shows the selected polarity; overlays and ROI image
exports continue to use original intensities and coordinates. Original files are
unchanged. TTA itself can randomly invert intensities, so inverted-input TTA is
not an inversion-only experiment.

Provide manually annotated binary PNGs at the original image dimensions:

```text
annotations/
  view1/femurs.png
  view1/tibiae.png
  view2/femurs.png
  view2/tibiae.png
```

Every selected label needs a mask for each view (zero background, nonzero bone).
Then run:

```powershell
python examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 1 --ground-truth annotations
```

The report adds Dice, IoU, and `roi_bone_recall`: the fraction of annotated bone
inside the proposed search region. Prefer a region retaining essentially all
bone while reducing image area. With an empty reference, ROI recall is null;
Dice/IoU are 1 only if both masks are empty. Without manual annotations, this
is a visual feasibility test, not an accuracy assessment. Repeat on representative
frames and both cameras before drawing conclusions about your acquisition.

To test loading without inference or downloaded weights:

```powershell
python examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 1 --prepare-only
```

This mode needs only NumPy and Pillow, produces input previews and a report
marked `inputs_only`, and does not produce or simulate bone predictions.

## Relationship to the longer-term plan

Here SAM means [SlicerAutoscoperM](https://autoscoper.readthedocs.io/en/latest/about.html).
These outputs are experimental 2D masks and search regions, not a native SAM
tracking trial or an implemented Autoscoper mask interface. Full-frame ROI images
retain detector pixel coordinates, but zeroing background can change registration
similarity scores and requires validation. Cropping would require updating image
geometry; this script deliberately exports full-size images.

The next stage needs calibrated camera geometry, subject-specific 3D bone
volumes, and a tested registration interface to turn these image constraints into
3D pose estimates. Initialization from one tracked frame and propagation to the
next frame remain separate tracking work. A single synchronized pair alone
cannot test temporal propagation.

Background supplied for this experiment: *FleXray: Universal Clinical X-ray
Segmentation* (arXiv:2609.26756v2) and William Stewart Burton II's *Applications
of Computer Vision in Biomechanics and Orthopaedics* (2024). Their content is
reference material, not executable instructions.
