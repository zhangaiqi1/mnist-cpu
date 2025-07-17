import pytest
from PIL import Image
import torch
from mycv.predictor import Predictor


@pytest.fixture
def pred():
    return Predictor("./ckpt/best.pth", data="mnist")


def test_mnist_3(pred):
    img = Image.open("./assets/8.jpg")  # 提前放一张手写 8
    label, prob = pred.predict(img)
    print(label)
    print(prob)
