from pathlib import Path
from PIL import Image
import argparse
from mycv.predictor import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--ckpt', default='ckpt/best.pth')
    parser.add_argument('--data', default='mnist')
    args = parser.parse_args()

    pred = Predictor(args.ckpt, data=args.data)
    for p in args.folder.glob('*'):
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            label, prob = pred.predict(Image.open(p))
            print(f"{p.name} -> {label} ({prob:.3f})")


if __name__ == "__main__":
    main()
