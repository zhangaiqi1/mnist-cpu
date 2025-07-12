import torch, os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_ckpt, load_ckpt


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, run_dir):
        self.model, self.opt, self.loss_fn = model, optimizer, loss_fn
        self.device = device
        self.writer = SummaryWriter(log_dir=run_dir)
        self.global_step = 0

    def fit(self, train_loader, val_loader, epochs, ckpt_dir, resume=None):
        start_epoch = best_acc = 0
        if resume and os.path.isfile(resume):  # 断点恢复逻辑
            start_epoch, best_acc = load_ckpt(resume, self.model, self.opt)
            print(f'Resume from {resume}, epoch {start_epoch}, best_acc={best_acc:.4f}')

        for epoch in range(start_epoch + 1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_acc = self._validate(val_loader, epoch)
            print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}')

            # 保存最好模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_ckpt(self.model, self.opt, epoch, best_acc,
                          os.path.join(ckpt_dir, 'best.pth'))
            # 同时保存最新
            save_ckpt(self.model, self.opt, epoch, best_acc,
                      os.path.join(ckpt_dir, 'last.pth'))

    def _train_one_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f'Train {epoch}')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.opt.step()
            running_loss += loss.item()
            pbar.set_postfix(
                loss=loss.item())  # 这是给 tqdm 进度条 动态追加显示字段的写法。
            # 例如 Epoch 1: 100%|██████████| 938/938 [00:21<00:00, 43.50it/s, loss=0.1234]
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.global_step += 1
        return running_loss / len(loader)

    @torch.no_grad()
    def _validate(self, loader, epoch):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / total
        self.writer.add_scalar('Acc/val', acc, epoch)
        return acc
