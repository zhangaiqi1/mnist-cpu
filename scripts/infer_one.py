from pathlib import Path
from PIL import Image
import argparse
from mycv.predictor import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', default="assets/8.jpg", type=Path)
    parser.add_argument('--ckpt', default='ckpt/best.pth')
    parser.add_argument('--data', default='mnist')
    args = parser.parse_args()

    pred = Predictor(args.ckpt, data=args.data)
    img = Image.open(args.img)
    label, prob = pred.predict(img)
    print(f"Predicted: {label}  confidence: {prob:.3f}")


if __name__ == "__main__":
    main()
