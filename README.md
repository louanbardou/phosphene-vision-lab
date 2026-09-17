# Phosphene Vision Simulator

A course exercise that visualizes what a cortical visual prosthesis might show
its wearer, starting from an ordinary photo. The pipeline uses only frozen,
pretrained/published weights — no training happens in this app. A photo is
encoded by a frozen self-supervised vision model, passed through frozen
published brain-response weights restricted to primary visual cortex (V1),
and the resulting target response is used to optimize a phosphene stimulation
pattern that is rendered through a real biologically-calibrated phosphene
simulator.

## Pipeline

1. **Encoder.** The input photo is passed through a frozen SimCLR ResNet18
   (`DM-Diaz/VEDB-SimCLR-ResNet18-Baseline` on HuggingFace), pretrained with
   self-supervised contrastive learning on the Visual Experience Dataset
   (VEDB), a large corpus of head-mounted, everyday first-person video. The
   encoder's weights are not modified anywhere in this app.

2. **Voxel encoding.** The resulting image features are mapped to predicted
   fMRI voxel responses using frozen ridge-regression weights
   (`DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models`), fit by the original authors
   on the Natural Scenes Dataset (NSD), a large public fMRI dataset of
   subjects viewing natural images. The predicted voxels are restricted to
   V1 using NSD's own public region-of-interest masks, downloaded directly
   from the NSD S3 bucket over anonymous HTTPS.

3. **Stimulation optimization.** With the encoder and ridge weights held
   completely fixed, a gradient-based, activation-maximization-style
   optimization searches for a stimulation pattern whose predicted V1
   response matches the target from step 2 as closely as possible. Only the
   stimulation pattern is optimized — no model weights change during this
   step.

4. **Phosphene rendering.** The optimized stimulation pattern is rendered
   into a simulated phosphene percept using
   [`dynaphos`](https://github.com/neuralcodinglab/dynaphos), a real,
   differentiable, biologically-calibrated simulator of cortical phosphene
   vision (Grinten, de Ruyter van Steveninck et al., *eLife* 2024). The
   electrode-to-cortex geometry comes from dynaphos's own coordinate-mapping
   utilities.

The visual layout is loosely inspired by CNN Explainer's clean,
step-by-step presentation — this project is not affiliated with CNN
Explainer and does not reuse its content.

## Honest limitations

- **PCA substitution.** The original authors reduced SimCLR features with a
  fitted PCA step before ridge regression. That fitted PCA object is not
  publicly distributed — this was checked against both the HuggingFace repo's
  file tree and the companion reproducibility code at
  [`DM-Diaz/eccentricity-constrained-simclr`](https://github.com/DM-Diaz/eccentricity-constrained-simclr),
  which shows how the PCA was fit but does not ship the fitted parameters.
  This app substitutes a fixed random orthogonal projection in its place,
  then applies the real, unmodified published ridge weights on top. The
  predicted "V1 target response" in this app should be read as illustrating
  how the pipeline mechanism works, not as a faithful reproduction of the
  paper's exact voxel predictions.
- **Per-subject weights, arbitrary photo.** The NSD ridge weights are fit per
  subject (S1-S8) on that individual's real fMRI data. Applying them to an
  arbitrary new photo from a different person is an approximation of how a
  generic visual cortex might respond, not a personalized readout of any
  particular viewer's own visual cortex.
- **Assumed electrode grid.** The electrode layout used for phosphene
  rendering is a regular grid assumed over a cortical map (via dynaphos's own
  coordinate-mapping utilities), not a measured or real electrode placement
  from an implanted device.
- **Not real-time video.** On CPU, the optimization takes a few seconds per
  still photo. This app intentionally works on single photos, one at a time,
  and does not attempt continuous live-video processing.

## Running locally

```bash
python3.11 -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
python setup_assets.py      # optional pre-download of default model/data assets
streamlit run app.py
```

`setup_assets.py` pre-downloads the default subject/model files into
`assets_cache/` so the first `streamlit run` doesn't stall on downloads. It's
optional — the app will fetch what it needs on demand if you skip this step.

## License

`dynaphos` is used as an unmodified pip dependency and is licensed GPL-3.0.
This project's own code has no license file yet — it's a course exercise.
