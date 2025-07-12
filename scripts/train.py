import torch, argparse, os
from mycv.datasets import get_loader
from mycv.models import build_model
from mycv.trainer import Trainer
from mycv.utils import make_run_dir
from torchvision.models import resnet18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist', choices=['mnist', 'cifar10', 'folder'])
    parser.add_argument('--root', default='./data')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--resume', default='')  # 断点路径
    args = parser.parse_args()

    device = torch.device('cpu')
    train_loader = get_loader(args.root, args.data, args.batch_size, train=True)
    val_loader = get_loader(args.root, args.data, args.batch_size, train=False)

    # 根据数据集选输入通道
    in_ch = 1 if args.data == 'mnist' else 3
    num_classes = 10 if args.data != 'folder' else 2
    model = build_model(args.model, in_ch, num_classes=num_classes).to(device)

    opt = SGD(model.parameters(), lr=args.lr, momentum=0.9)  # 带有动量的SGD
    loss_fn = CrossEntropyLoss()

    run_dir = make_run_dir('runs')
    ckpt_dir = 'ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Trainer(model, opt, loss_fn, device, run_dir)
    trainer.fit(train_loader, val_loader, args.epochs, ckpt_dir, resume=args.resume)


if __name__ == '__main__':
    main()
