import argparse, json
from pathlib import Path
from PIL import Image
from src.sketch2vision import Sketch2Vision

def main():
    parser = argparse.ArgumentParser(description="ControlNet strength ablation for all three personal sketches")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.45, 0.75, 1.05])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    entries = json.loads((root / "data/prompts.json").read_text())
    out = root / "outputs/strength_sweep"
    out.mkdir(parents=True, exist_ok=True)
    generator = Sketch2Vision()
    records = []
    for key, item in entries.items():
        sketch = Image.open(root / item["input"]).convert("RGB")
        for scale in args.scales:
            image, _ = generator.generate(
                sketch, item["prompt"], item["negative_prompt"],
                scale, args.steps, args.guidance, args.seed, args.size
            )
            name = f"{key}_strength_{scale:.2f}.png"
            image.save(out / name)
            records.append({"id": key, "strength": scale, "file": name,
                            "prompt": item["prompt"], "seed": args.seed,
                            "steps": args.steps, "guidance": args.guidance})
            print(f"Saved {out / name}")
    (out / "experiment_manifest.json").write_text(json.dumps(records, indent=2))
if __name__ == "__main__":
    main()
