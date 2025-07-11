import torch, os, time
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model, self.opt, self.loss_fn = model, optimizer, loss_fn
        self.device = device

    def fit(self, train_loader, val_loader, epochs, ckpt_path='ckpt.pt'):
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for x, y in tqdm(train_loader, desc=f'Epoch {epoch}'):
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                out = self.model(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()
            val_acc = self.evaluate(val_loader)
            print(f'Epoch {epoch}: loss={running_loss:.4f} val_acc={val_acc:.4f}')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total
