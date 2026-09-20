# Phosphene Vision Lab

Final project for Tooling for the Data Scientist. Turns a photo into a simulated view of what a cortical visual prosthesis wearer might see.

## Pipeline

1. Photo → frozen SimCLR ResNet18 (`DM-Diaz/VEDB-SimCLR-ResNet18-Baseline` on HuggingFace).
2. Features → frozen NSD ridge-regression weights (`DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models`), kept to V1 voxels only using NSD's public ROI masks.
3. Gradient descent on the stimulation pattern only (nothing gets trained) to match that V1 target.
4. Rendered through [`dynaphos`](https://github.com/neuralcodinglab/dynaphos), a real phosphene simulator. The output is the "percept" — what the prosthesis wearer would subjectively see, not a brain scan.

## Limitations

- No public PCA basis for the feature reduction step → replaced with a fixed random projection. V1 target = illustrative, not the paper's exact numbers.
- Ridge weights are fit per subject (S1-S8), not "your" brain.
- Electrode grid is assumed (regular grid), not a real implant layout.
- Runs on CPU in a few seconds per photo, not live video.

## Run it

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the model/data files (~250MB, public, no login needed).

## Tests

Unit tests cover the data import/filtering functions (picking the right weight file, filtering V1 voxels, placing values back in the brain volume) with small fake inputs — no network calls, so they're fast and don't break in CI.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the test suite and builds the Docker image on every push/PR.

## Docker

```bash
docker build -t phosphene-vision-lab .
docker run -p 8501:8501 phosphene-vision-lab
```

Then open http://localhost:8501.
