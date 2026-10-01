# FleXray / SlicerAutoscoperM project notes

Updated: 1 October 2026. Copy this page into OneNote.

## Aim

Use FleXray to identify femur and tibia regions in paired knee X-rays, inspect masked trials in SlicerAutoscoperM (SAM), then evaluate whether those regions can improve tracking. Start by testing the masked export in MATLAB; a Python equivalent can follow after the behaviour is validated.

## Work completed so far

- Established single-pair and sequence processing for two X-ray cameras, with separate femur and tibia predictions.
- Added original-coordinate masks, probabilities, overlays, per-frame reports, sequence review pages and a combined camera video.
- Explored intensity inversion, a leg crop, largest-component cleanup, detector-boundary exclusion, hole filling and per-camera orientation correction. These are experimental options rather than proven settings for all trials.
- Added MATLAB exports for femur-only, tibia-only and combined masked X-rays, with adjustable dilation and a copied SAM configuration for each version. Image dimensions and retained pixel intensities are preserved.
- Found an empty tibia prediction at source frame 89 in camera 1 of the KneeFlex sequence. Changed the exporter to keep an empty frame and record it, rather than stop or misalign cameras.
- Added selectable roots, activity/trial selection, automatic frame-range discovery and user-selected output destinations. Added a dry-run option to check Python selection before inference.
- Organised MATLAB implementations under examples/matlab, retained the old launcher commands and removed personal machine paths from the workflow defaults and setup examples.

## Current status and limits

- Verification on 1 October: 16 focused offline Python tests passed. A real-data dry run found six activities and 287 KneeFlex frame pairs (1-287); it did not run inference. Offline tests used the available Anaconda runtime, not a fresh supported inference installation.
- Existing saved outputs demonstrate segmentation runs, but no manual ground-truth comparison or measured pose improvement has been established.
- Predictions are independent per frame; this is not temporal tracking or propagation.
- MATLAB export code exists, but the earlier automated MATLAB run was blocked by the university licence server. Successful loading and tracking of exported trials in SAM still need to be demonstrated.
- A blank background is not a verified exclusion mask for SAM's tracking objective. Mask boundaries may themselves affect registration.
- Both bone-model entries remain in the copied SAM configurations. Each separate export controls image visibility, not model selection.
- The supplied StepDown configuration names trial 04, while some source folders name trial 03. Use the matching acquisition and segmentation output; the exporter checks the trial folder names.
- Matching camera frame numbers are assumed to indicate synchronization and should be verified.

## Immediate next actions

- [ ] Run the selection dry-run on each activity; check the chosen trial/session and reported frame count.
- [ ] Repair/select a supported local Python environment for model inference and confirm MATLAB licence access.
- [ ] Review representative beginning, middle and end frames in both cameras, including frame 89 and orientation transitions.
- [ ] Export a short contiguous range in MATLAB and open all three configurations in SAM.
- [ ] Verify image orientation, original dimensions, camera geometry, intensity appearance and frame numbering against the unmasked trial.
- [ ] Compare dilation values (for example 0, 10, 20 and 30 pixels) and record whether bone boundaries remain included.
- [ ] Record missing or poor masks explicitly; avoid treating blank frames as usable tracking evidence.

## Next stage: evaluate tracking

- [ ] Establish how SAM can accept a region/weight mask and whether background pixels are actually excluded from the similarity calculation.
- [ ] Compare the original and masked trials using the same initialization, frame range and tracking settings.
- [ ] Measure tracking success, pose differences, convergence and runtime; inspect failures and artificial boundary effects.
- [ ] Create manual masks for representative frames to measure Dice/IoU and retained bone coverage, including difficult views and activities.
- [ ] Decide whether separate bone tracking or a combined region is appropriate after observing the SAM results.

## Later development

- [ ] Port the validated MATLAB export behaviour to Python, including separate bones, dilation, frame-status logging and configuration generation.
- [ ] Add portable path handling for shared trial packages and define required calibration/model assets.
- [ ] Consider temporal consistency checks or propagation only after frame-level performance is characterised.
- [ ] Run a collaborator trial from a fresh environment using GETTING_STARTED.md; record any missing setup steps.

## Evidence to capture for the next review

Record activity/trial, source and mask frame range, model revision, preprocessing settings, dilation radius, empty-mask counts, screenshots of both cameras in SAM, and the tracking comparison results. Keep segmentation quality and tracking quality as separate outcomes.
