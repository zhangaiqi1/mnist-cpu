"""
放两个小工具：日志目录、断点保存/加载

"""

import os, torch, time
from pathlib import Path


def make_run_dir(root="./runs"):
    run_id = time.strftime('%Y%m%d_%H%M%S')
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_ckpt(model, optimizer, epoch, best_acc, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
    }, path)


def load_ckpt(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['best_acc']
