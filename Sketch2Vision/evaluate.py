import argparse, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def edge_f1(sketch_path, generated_path, size=512, tolerance=3):
    def edges(path):
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(im, ((size-im.width)//2, (size-im.height)//2))
        gray = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        return cv2.Canny(gray, 60, 150) > 0
    a, b = edges(sketch_path), edges(generated_path)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    a_d = cv2.dilate(a.astype(np.uint8), kernel) > 0
    b_d = cv2.dilate(b.astype(np.uint8), kernel) > 0
    precision = float((b & a_d).sum() / max(b.sum(), 1))
    recall = float((a & b_d).sum() / max(a.sum(), 1))
    f1 = 2*precision*recall/max(precision+recall, 1e-12)
    return precision, recall, f1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/baseline/run_manifest.json")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / args.manifest).read_text())
    print("id,precision,recall,edge_f1")
    for item in manifest["results"]:
        vals = edge_f1(root / item["sketch"], root / item["generated"])
        print(f'{item["id"]},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f}')
if __name__ == "__main__":
    main()
