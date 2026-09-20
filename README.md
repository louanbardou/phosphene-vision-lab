# Phosphene Vision Lab

Final project for Tooling for the Data Scientist.

This app shows what a blind person with a brain implant (a "visual prosthesis") might see when looking at a photo. You upload a photo, and it simulates the pattern of light spots ("phosphenes") that person's brain implant would probably produce.

## Background

The idea started at UCSF during an internship on deep brain stimulation (DBS) for treatment-resistant depression. DBS uses a different implant and targets a different brain circuit, but that internship is where I got curious about designing and testing brain stimulation patterns computationally before they reach a real device. This project turns that curiosity toward a cortical visual prosthesis, as the final project for Tooling for the Data Scientist.

## How it works (simple version)

1. An AI model trained to recognize images looks at your photo.
2. Another model, trained on real brain-scan data, guesses how a person's brain would react to it.
3. A search program tries many electrical patterns until it finds one that would cause a similar brain reaction.
4. A simulator shows what that pattern would look like as spots of light. This is the "percept": what the person would see.

We didn't train anything here; every model is public and pretrained. Step 3 is the only part doing real computing, searching for the best pattern.

## Good to know

- One step (turning image features into a smaller set of numbers) is missing its original recipe, so we swapped in a simple substitute. The "brain reaction" shown is a rough illustration; it doesn't match the original research exactly.
- The brain data comes from one of 8 real people (S1-S8) who took part in a real study. It isn't your brain.
- The electrode positions are a made-up regular grid. No real implant is laid out this way.
- It takes a few seconds per photo on a normal computer's CPU, so it won't keep up with real-time video.

## Run it

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the model/data files (~250MB, public, no login needed).

## Tests

Unit tests check the functions that pick/filter the data (picking the right file, keeping only the right brain region, placing values back in the right spot). They run on small fake inputs instead of the real downloads, so they're fast and don't need internet.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src.pipeline --cov-report=term-missing
```

Coverage is around 30% of `src/pipeline.py`. The untested part is mostly the download/model-loading code, since it needs internet and real files. I checked that part by hand with the smoke test in that file.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the test suite and builds the Docker image on every push/PR.

## Docker

```bash
docker build -t phosphene-vision-lab .
docker run -p 8501:8501 phosphene-vision-lab
```

Then open http://localhost:8501.
