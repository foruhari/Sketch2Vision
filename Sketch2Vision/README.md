# Sketch2Vision — Personal Hand-Drawn Sketch-to-Image Project

**Machine Learning II · Individual project · 20 November 2026**

This repository implements sketch-conditioned image generation with a pretrained Stable Diffusion 1.5 model and Canny ControlNet. The three inputs are the student's own hand-drawn pencil sketches, photographed and cropped from the supplied images:
1. Prague cityscape on a mirror
2. Brugge cityscape coaster
3. Cowboy cactus painting

The original photographs are retained in `data/references/` for qualitative comparison. Each sketch is a separate input file in `data/sketches/`. `sketches_contact_sheet.jpg` is only a visual overview; do not upload it as a single model input.

## Method and scope

- Base model: `runwayml/stable-diffusion-v1-5`
- Control model: `lllyasviel/sd-controlnet-canny`
- Framework: PyTorch + Hugging Face Diffusers
- Input processing: crop/resize, denoise, contrast enhancement, Canny edge extraction
- UI: Streamlit
- Experiment: hold sketch, prompt, seed, steps and CFG constant; vary ControlNet strength (0.45, 0.75, 1.05)
- Evaluation: tolerant edge-overlap precision/recall/F1 as a limited structural proxy, plus qualitative comparison

The pretrained models are used for inference. **No model training or fine-tuning is required** by this implementation. The technical work is the preprocessing, conditioning pipeline, UI, controlled experiment, evaluation and reproducibility.

## What is already prepared

- Three separate personal pencil sketches.
- Three corresponding original reference photographs.
- Three prompts and negative prompts in `data/prompts.json`.
- Reproducible inference and experiment scripts.
- A Streamlit application.
- An editable report draft in `docs/REPORT.md`.

## Requirements

Python 3.10 or 3.11; internet access for initial model download; several GB disk space. An NVIDIA GPU is strongly recommended (about 10–12 GB VRAM or more). If you do not have a compatible GPU, use Google Colab with a GPU runtime. CPU generation is possible in principle but may be extremely slow.

Hugging Face is used as the model repository/library source; the assignment does not require publishing the project on Hugging Face. Publish the code on GitHub.

## Run locally (Linux / Ubuntu)

```bash
cd Sketch2Vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if needed, following https://pytorch.org/get-started/locally/.

Launch the interface:

```bash
streamlit run app.py
```

Upload one sketch at a time from `data/sketches/`. Use its matching prompt from `data/prompts.json`. Suggested starting settings: 512×512, 30 steps, CFG 7.5, seed 123, ControlNet strength 0.75.

The first run downloads several GB of model weights and can take time. Model access and license terms can change; read the model cards before use.

## Run generation from the command line

Generate all three baseline results:

```bash
python generate_examples.py --steps 30 --guidance 7.5 --strength 0.75 --seed 123
```

Outputs are written to `outputs/baseline/`, with a `run_manifest.json` recording prompts and settings.

## Run the controlled experiment

```bash
python experiment_strength.py --steps 30 --guidance 7.5 --seed 123 --scales 0.45 0.75 1.05
```

The script runs each of the three sketches at each strength. Only ControlNet strength changes within each three-image sweep. This is a 3-sketch × 3-strength experiment (9 outputs).

## Evaluate sketch-structure preservation

After generation:

```bash
python evaluate.py --manifest outputs/baseline/run_manifest.json
```

The edge F1 score is a rough structural proxy. It is not a semantic or perceptual quality metric. Interpret it alongside the side-by-side figures; generated texture and fine detail can lower the score even when the main object is recognizable.

## Suggested workflow / submission checklist

1. Run the environment setup and check that PyTorch detects the GPU.
2. Generate the three baseline outputs.
3. Run the nine-image conditioning-strength experiment.
4. Inspect outputs and record honest observations; do not invent scores.
5. Insert generated figures and actual metrics into `docs/REPORT.md`.
6. Export the final report to PDF (2–4 pages, per assignment brief).
7. Push the code, input sketches, prompts, outputs selected for the report and report to GitHub. Do not commit model weight files or `.venv/`.

## References

- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. https://arxiv.org/abs/2006.11239
- Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. https://arxiv.org/abs/2112.10752
- Zhang, L. & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. https://arxiv.org/abs/2302.05543
- Stable Diffusion v1.5 model card: https://huggingface.co/runwayml/stable-diffusion-v1-5
- ControlNet Canny model card: https://huggingface.co/lllyasviel/sd-controlnet-canny
- Diffusers ControlNet documentation: https://huggingface.co/docs/diffusers/using-diffusers/controlnet

## Course connection

Session 2 introduces diffusion as progressive noising followed by learned reverse denoising, and latent diffusion as operating in a compressed latent representation. Its dermatology case study combines Stable Diffusion with ControlNet for text/sketch conditioning. This project applies the same broad conditioning concept to personal creative sketches, without reproducing the medical application or fine-tuning its model.
