import torch, torchvision.transforms as T
from pathlib import Path
from PIL import Image
from mycv.models import build_model


# 新增量化开关
class Predictor:
    def __init__(self, ckpt_path: str, model_name: str = "resnet18", data: str = "mnist", quantized=False):
        self.device = torch.device("cpu")
        # 根据数据集选通道
        in_ch = 1 if data == "mnist" else 3
        num_classes = 10 if data != "folder" else 2
        self.model = build_model(model_name, in_ch, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu")['model'])
        self.model.eval()

        # 统一的预处理
        if data == "mnist":
            self.transform = T.Compose([
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))
            ])
        else:  # cifar10 / folder
            self.transform = T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
            ])
        if quantized:
            from torch.ao.quantization import quantize_dynamic
            self.model = quantize_dynamic(self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
        self.model.eval()

    def predict(self, img: Image.Image):
        """
        输入 PIL.Image → 返回 (class_id, prob)
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1).squeeze(0)
            pred = prob.argmax().item()
        return pred, prob[pred].item()
