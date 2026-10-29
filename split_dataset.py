"""Split data/raw/<class>/ into data/currency/train and data/currency/val (copies files)."""
import argparse
import os
import random
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw")
    p.add_argument("--dst", default="data/currency")
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    for cls in sorted(os.listdir(args.src)):
        src_dir = os.path.join(args.src, cls)
        if not os.path.isdir(src_dir):
            continue
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val))
        for split, names in (("val", files[:n_val]), ("train", files[n_val:])):
            out_dir = os.path.join(args.dst, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for name in names:
                shutil.copy2(os.path.join(src_dir, name), out_dir)
        print(f"{cls:>12}: {len(files) - n_val} train, {n_val} val")


if __name__ == "__main__":
    main()
