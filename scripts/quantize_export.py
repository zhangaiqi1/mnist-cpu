import torch, argparse, os
from mycv.models import build_model
from torch.ao.quantization import quantize_dynamic, QConfigMapping


def quantize_and_save(ckpt_fp32, ckpt_int8, data='mnist'):
    in_ch = 1 if data == 'mnist' else 3
    num_classes = 10 if data != 'folder' else 2
    model = build_model('resnet18', in_ch, num_classes)
    model.load_state_dict(torch.load(ckpt_fp32, map_location='cpu')['model'])
    model.eval()

    # 动态量化（仅 Linear 和 Conv）
    qmodel = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    torch.save({
        'model': qmodel.state_dict(),
    }, ckpt_int8)
    torch.save(qmodel.state_dict(), ckpt_int8)
    print(f"INT8 model saved -> {ckpt_int8}  "
          f"size={os.path.getsize(ckpt_int8) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_fp32', default='ckpt/best.pth')
    parser.add_argument('--ckpt_int8', default='ckpt/best_int8.pth')
    parser.add_argument('--data', default='mnist')
    args = parser.parse_args()
    quantize_and_save(args.ckpt_fp32, args.ckpt_int8, args.data)
