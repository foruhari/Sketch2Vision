# Sketch2Vision: Structure-Preserving Generation from Personal Hand-Drawn Sketches

**Machine Learning II · Individual project · 20 November 2026**  
**Student:** Maryam Forouhari

## Abstract
This project develops a reproducible sketch-to-image pipeline using a pretrained Stable Diffusion 1.5 model conditioned by a Canny ControlNet. Three personally hand-drawn pencil sketches are used as inputs: a Prague cityscape on a mirror, a Brugge souvenir coaster and a cowboy cactus illustration. Each sketch is processed into an edge condition and combined with a text prompt. The system generates a new image while attempting to preserve the sketch's main spatial structure. The implementation includes a Python inference pipeline, a Streamlit interface, a controlled ControlNet-strength experiment and an approximate edge-overlap evaluation. The pretrained models are used without fine-tuning.

## 1. Method
Diffusion models generate images by learning to reverse a gradual noising process. Latent diffusion performs denoising in a compressed latent representation, reducing computation compared with pixel-space diffusion. In this project, the text prompt describes the desired content and visual style, while ControlNet provides spatial guidance derived from the hand-drawn sketch. The ControlNet residuals guide the denoising U-Net so that generated content follows the sketch more closely.

**Models and tools:** Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`), Canny ControlNet (`lllyasviel/sd-controlnet-canny`), PyTorch, Hugging Face Diffusers, OpenCV, Pillow and Streamlit.

**Preprocessing:** each photographed sketch is cropped to its drawing area, resized with aspect ratio preserved, lightly denoised and contrast-enhanced. Canny edge detection creates the structural control image. The original photographs are retained as qualitative references; they are not used as model targets or training data.

## 2. Personal inputs and implementation
The three independent inputs are:
1. **Prague mirror:** a pencil drawing of the historic skyline, towers and bridge arches.
2. **Brugge coaster:** a pencil drawing of the souvenir design, central lettering, buildings and floral decoration.
3. **Cowboy cactus:** a pencil drawing of a cactus with curved arms, dotted spines, flowers and a cowboy hat.

The code in `src/sketch2vision.py` loads the pretrained models, preprocesses the sketch and runs conditioned generation. `app.py` provides an interactive interface. `generate_examples.py` creates one baseline output for each input. `experiment_strength.py` varies ControlNet strength across 0.45, 0.75 and 1.05 while holding the prompt, seed, number of steps and guidance constant. This gives nine experiment outputs. `evaluate.py` computes tolerant edge precision, recall and F1 as a rough structural-preservation proxy.

## 3. Experimental protocol and results
Baseline settings: 512 × 512 output, 30 denoising steps, CFG 7.5, seed 123 and ControlNet strength 0.75. For the ablation, strength values are 0.45, 0.75 and 1.05; all other settings remain fixed. Each sketch has its own descriptive prompt in `data/prompts.json`.

**Results must be populated after running the code. No generated images or numerical results are claimed in this draft.**

| Input sketch | Strength | Generated image | Edge F1 | Observation |
|---|---:|---|---:|---|
| Prague mirror | 0.45 / 0.75 / 1.05 | To be generated | To be measured | To be completed |
| Brugge coaster | 0.45 / 0.75 / 1.05 | To be generated | To be measured | To be completed |
| Cowboy cactus | 0.45 / 0.75 / 1.05 | To be generated | To be measured | To be completed |

The edge F1 metric measures proximity between edges in the input sketch and generated image. It is only a structural proxy: extra texture and fine detail can lower precision, and it does not measure semantic correctness, aesthetic quality or exact object identity. Therefore, numerical scores must be discussed together with side-by-side visual inspection.

**Figures to insert after execution:** for each input, show (a) the hand-drawn sketch, (b) its Canny control image and (c) generated output. For the experiment, include a three-way comparison of strength 0.45, 0.75 and 1.05. Caption every figure with prompt, seed and generation settings.

## 4. Discussion and limitations
ControlNet strength controls the trade-off between structural adherence and generative freedom. Lower values allow the text prompt more influence and may alter layout or proportions. Higher values emphasize the input edges but may preserve unwanted marks or constrain visual detail. The optimal setting may differ between the architectural, decorative and illustrative sketches. The experiment isolates this parameter to make the comparison interpretable.

Limitations include shadows and paper texture in photographed sketches, imperfect Canny edge extraction, text rendering errors (especially the word “BRUGGE”), model bias and imperfect geometry preservation. The edge-overlap metric is not a complete measure of quality. The model also requires substantial compute; a CUDA GPU or hosted GPU runtime is recommended.

## 5. Conclusion
The project demonstrates a technical sketch-to-image system using personal hand-drawn inputs, text prompts and ControlNet conditioning. It includes preprocessing, reproducible inference, a user interface, a controlled conditioning-strength experiment and a structural proxy metric. The work uses pretrained models rather than training from scratch or fine-tuning. Future work could compare Canny with a scribble-specific ControlNet, add human evaluation or train a lightweight LoRA on a suitable licensed dataset.

## References
1. Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. https://arxiv.org/abs/2006.11239
2. Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. https://arxiv.org/abs/2112.10752
3. Zhang, L. & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. https://arxiv.org/abs/2302.05543
4. Stable Diffusion v1.5 model card: https://huggingface.co/runwayml/stable-diffusion-v1-5
5. ControlNet Canny model card: https://huggingface.co/lllyasviel/sd-controlnet-canny
6. Hugging Face Diffusers documentation: https://huggingface.co/docs/diffusers/using-diffusers/controlnet
7. Machine Learning II, Session 2 — Generative AI; Session 4 — Modern AI.
