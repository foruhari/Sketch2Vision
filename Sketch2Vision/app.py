import streamlit as st
from PIL import Image
from pathlib import Path
import sys, io
sys.path.insert(0, str(Path(__file__).parent / "src"))
from sketch2vision import Sketch2Vision, prepare_sketch

st.set_page_config(page_title="Sketch2Vision", page_icon="✏️", layout="wide")
st.title("✏️ Sketch2Vision")
st.caption("Turn a hand-drawn sketch into a detailed image while conditioning generation on its structure.")
st.info("The first run downloads pretrained model weights (several GB). A CUDA GPU is strongly recommended.")
uploaded = st.file_uploader("Upload a sketch (PNG/JPG)", type=["png","jpg","jpeg"])
left, right = st.columns(2)
with left:
    prompt = st.text_area("Describe the desired result", "A photorealistic futuristic sustainable building, glass and brushed metal facade, warm golden-hour lighting, architectural photography, highly detailed")
    negative = st.text_input("Negative prompt", "blurry, low quality, distorted geometry, extra objects, text, watermark")
    style = st.selectbox("Style preset", ["Photorealistic", "Concept art", "Product render", "3D illustration", "Watercolor"])
    scale = st.slider("Sketch preservation (ControlNet strength)", 0.25, 1.50, 0.85, 0.05)
    steps = st.slider("Denoising steps", 15, 50, 30, 1)
    guidance = st.slider("Text guidance (CFG)", 3.0, 12.0, 7.5, 0.5)
    seed = st.number_input("Seed (same seed helps compare settings)", min_value=0, max_value=999999, value=42)
    size = st.selectbox("Output resolution", [512, 640, 768], index=0)
    run = st.button("Generate image", type="primary", disabled=uploaded is None)
with right:
    if uploaded:
        sketch = Image.open(uploaded).convert("RGB")
        st.image(sketch, caption="Original sketch", use_container_width=True)
        _, control = prepare_sketch(sketch, size=512)
        st.image(control, caption="Canny structural condition", use_container_width=True)
    else:
        st.write("Upload one of the example sketches from `data/sketches/` or your own drawing.")
if run and uploaded:
    styles = {
        "Photorealistic": "photorealistic, realistic materials, natural lighting",
        "Concept art": "professional concept art, detailed digital painting",
        "Product render": "premium studio product rendering, clean background, realistic materials",
        "3D illustration": "high quality 3D render, physically based materials",
        "Watercolor": "delicate watercolor illustration, textured paper"
    }
    full_prompt = f"{prompt}, {styles[style]}"
    with st.spinner("Loading models and generating… first run may take several minutes."):
        if "engine" not in st.session_state:
            st.session_state.engine = Sketch2Vision()
        output, condition = st.session_state.engine.generate(
            sketch, full_prompt, negative, scale, steps, guidance, int(seed), int(size)
        )
    st.success("Generation complete.")
    a,b,c = st.columns(3)
    a.image(sketch, caption="Input sketch", use_container_width=True)
    b.image(condition, caption="ControlNet condition", use_container_width=True)
    c.image(output, caption="Generated image", use_container_width=True)
    buf=io.BytesIO(); output.save(buf, format="PNG")
    st.download_button("Download generated PNG", buf.getvalue(), "sketch2vision_output.png", "image/png")
    st.caption(f"Prompt: {full_prompt} | seed={seed} | control={scale} | steps={steps}")
