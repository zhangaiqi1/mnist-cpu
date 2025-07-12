from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loader(root: str, name: str, batch_size: int, train: bool = True):
    """
    root: 数据集根目录
    name: 'mnist' | 'cifar10' | 'folder'
    train: True/False
    """
    if name == 'mnist':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 注意这里的，等会运行的时候验证
        ])
        dataset = datasets.MNIST(root, train=train, download=True, transform=trans)  # 下载
    elif name == 'cifar10':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]) # 注意这里的，等会运行的时候验证
        dataset = datasets.CIFAR10(root, train=train, download=True, transform=trans)
    elif name == 'folder':
        # 任意图像文件夹：root/train 或 root/val
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        sub = 'train' if train else 'val'
        dataset = datasets.ImageFolder(root=f"{root}/{sub}", transform=trans)
    else:
        raise ValueError(name)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=train, num_workers=0)  # CPU 设 0
