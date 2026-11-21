import argparse, json
from pathlib import Path
from PIL import Image
from src.sketch2vision import Sketch2Vision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    entries = json.loads((root / "data/prompts.json").read_text())
    out = root / "outputs/baseline"
    out.mkdir(parents=True, exist_ok=True)
    generator = Sketch2Vision()
    manifest = {"settings": vars(args), "model": "runwayml/stable-diffusion-v1-5",
                "controlnet": "lllyasviel/sd-controlnet-canny", "results": []}
    for key, item in entries.items():
        sketch_path = root / item["input"]
        sketch = Image.open(sketch_path).convert("RGB")
        image, control = generator.generate(
            sketch, item["prompt"], item["negative_prompt"],
            args.strength, args.steps, args.guidance, args.seed, args.size
        )
        image_path = out / f"{key}_generated.png"
        control_path = out / f"{key}_control.png"
        image.save(image_path)
        control.save(control_path)
        manifest["results"].append({
            "id": key, "sketch": item["input"],
            "reference": item["reference"],
            "generated": str(image_path.relative_to(root)),
            "control": str(control_path.relative_to(root)),
            "prompt": item["prompt"], "negative_prompt": item["negative_prompt"]
        })
        print(f"Saved {image_path}")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {out / 'run_manifest.json'}")

if __name__ == "__main__":
    main()
