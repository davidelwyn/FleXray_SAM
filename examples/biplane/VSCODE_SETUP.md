# Windows / VS Code setup

## 1. Python and VS Code

Use 64-bit Python 3.11 for this setup. FleXray requires Python >=3.10;
Python 3.9 is too old. Install a suitable interpreter from the
[official Windows downloads](https://www.python.org/downloads/windows/) or
create a Python 3.11 environment with your existing Conda installation.

Install Microsoft's **Python** extension in VS Code. Open this folder with
**File > Open Folder**:

```text
D:\Projects\FleXray_SAM
```

## 2. Create an isolated environment and install

Open **Terminal > New Terminal** (PowerShell), then run:

```powershell
cd D:\Projects\FleXray_SAM
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
```

If `py -3.11` is unavailable, Python 3.11 is not registered with the launcher;
use the full path to that interpreter for the environment creation command.
These commands call the environment's Python directly, so PowerShell script
activation and execution-policy changes are unnecessary.

`pip install -e .` installs the local repository and all required dependencies
listed in `pyproject.toml`. You do not need separate commands for each package:

| Package | Required version | Purpose |
| --- | --- | --- |
| torch | >=2.3,<3 | Model inference |
| torchvision | >=0.18,<1 | PyTorch image/model utilities |
| numpy | >=1.23 | Arrays and exported probabilities |
| Pillow | >=10 | TIFF/PNG input, masks, previews |
| scipy | >=1.10 | FleXray scientific utilities |
| huggingface_hub | >=0.24 | Fetch pretrained model bundles |
| imageio | >=2.34 | Write the combined biplane MP4 |
| imageio-ffmpeg | >=0.5 | Bundled video encoder used by imageio |
| safetensors | >=0.4 | Read model weights |
| pyyaml | >=6.0 | Model configuration |

Pip also resolves these packages' own dependencies. A CPU run is sufficient for
the initial test; it can be slower than GPU inference. `--device auto` uses CUDA
only when the installed PyTorch build and hardware support it. Check with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For GPU installation, follow the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
for your hardware and driver, retaining this repository's version constraints.
Do not install training extras for this test. Slicer, Autoscoper, Segment
Anything, Jupyter, and OpenCV are not required to run this standalone script.

## 3. Select the interpreter

Press **Ctrl+Shift+P**, run **Python: Select Interpreter**, and select
`FleXray_SAM\.venv\Scripts\python.exe`. If it is missing from the list, use
**Enter interpreter path**. See the official
[VS Code environment guide](https://code.visualstudio.com/docs/python/environments).

## 4. Check the inputs, then run the model

```powershell
.\.venv\Scripts\python.exe examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 1 --prepare-only
.\.venv\Scripts\python.exe examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 1 --device cpu
```

The second command downloads model weights on its first run and predicts femur
and tibia masks for both camera views. Internet access is needed for that first
download; the X-ray arrays are processed locally. To use an existing model
bundle, add `--model "C:\path\to\bundle"`.

To try more knee labels or another frame:

```powershell
.\.venv\Scripts\python.exe examples/biplane/test_pair.py --data-dir "D:\Xrays\one_trial" --frame 25 --labels femurs tibiae patellae fibulae --margin 20 --device auto
```

The commands above use the directory supplied with `--data-dir`. Open
the new `outputs/biplane/<timestamp>/comparison.png` in VS Code. `report.json`
contains per-bone search-area reduction and detection status. See
[README.md](README.md) for explicit image selection, annotation scoring, and
the limitations of this first-stage experiment.

## 5. Optional checks

The focused offline tests need no additional testing dependency:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_biplane_example tests.test_biplane_sequence -v
```

For pytest instead:

```powershell
.\.venv\Scripts\python.exe -m pip install "pytest>=8"
.\.venv\Scripts\python.exe -m pytest tests/test_biplane_example.py -q
```

## Troubleshooting

- `No module named torch` or `fxr`: rerun the editable install using the exact
  `.venv\Scripts\python.exe` above and check the VS Code interpreter.
- Model download fails: check access to Hugging Face or use a local bundle.
- CUDA errors: add `--device cpu` for the initial test.
- Missing or ambiguous camera files: supply `--view1` and `--view2` explicitly;
  paths are relative to `--data-dir` unless absolute.
- Existing output folder: choose a new `--output` path. Sequence `--interactive`
  creates a timestamped run inside your selected output parent.
- Empty or poor masks: inspect input previews, threshold and margin; the script
  flags empty detections and retains the full frame. A completed run does not
  establish segmentation accuracy without manual reference masks.
