import tempfile, torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from mycv.models import build_model
from mycv.trainer import Trainer
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

def test_trainer_one_step():
    model = build_model('resnet18', 1, 10)
    opt = SGD(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    fake = FakeData(size=4, image_size=(1, 28, 28), num_classes=10, transform=torch.ToTensor())
    loader = DataLoader(fake, batch_size=2)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, opt, loss_fn, 'cpu', tmp)
        trainer.fit(loader, loader, epochs=1, ckpt_dir=tmp)