import torch, argparse, os
from mycv.datasets import get_loader
from torchvision.models import resnet18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist', choices=['mnist', 'cifar10', 'folder'])
    parser.add_argument('--root', default='./data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cpu')
    train_loader = get_loader(args.root, args.data, args.batch_size, train=True)
    val_loader = get_loader(args.root, args.data, args.batch_size, train=False)

    # 根据数据集选输入通道
    in_ch = 1 if args.data == 'mnist' else 3
    model = resnet18(num_classes=10 if args.data != 'folder' else 2)  # 用的是torch模型里定义好的resnet18
    # 原始的resnet第一个conv1的输入是3，所有这里需要修改
    model.conv1 = torch.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)

    opt = SGD(model.parameters(), lr=args.lr, momentum=0.9)  # 带有动量的SGD
    loss_fn = CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        running_loss = 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        # 快速验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f'Epoch {epoch}: train_loss={running_loss / len(train_loader):.4f} val_acc={acc:.4f}')


if __name__ == '__main__':
    main()
