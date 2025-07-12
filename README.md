# Day 1 – 2025-07-11

> 项目：mnist_cpu
---

### 项目结构

> mnist_cpu/  
> ├── mycv/  
> │ ├── __init__.py  
> │ ├── datasets.py  
> │ ├── models.py  
> │ ├── trainer.py  
> │ └── utils.py  
> ├── scripts/  
> │ ├── train.py  
> │ ├── eval.py  
> │ └── infer_one.py  
> ├── tests/  
> ├── requirements.txt  
> └── README.md

---

### ✅ 今日完成清单

| 任务             | 状态 | 备注                              |
|----------------|------|---------------------------------|
| 搭建项目文件夹模板      | ✅ | `/mycv` `/scriots` `/tests` 已建好 |
| 完成通用Trainer类模板 | ✅ | 支持训练、验证、自动保存最优模型                |

---

# Day 2 - 2025-07-12

---

### ✅ 今日完成清单

| 任务                        | 状态 | 备注                     |
|---------------------------|------|------------------------|
| 编写`mycv/trianer`的Trainer类 | ✅ | 有了该类，只需要在`main`里调用即可   |
| 编写`mycv/datasets`         | ✅ | 用于加载数据集                |
| 编写`mycv/models`           |✅| 存放模型的文件，后续切换模型，只需要修改这里 |
| 编写`mycv/utils`            |✅| 放两个小工具：日志目录、断点保存/加载    |
| 改写`scripts/train.py`      |✅| 改写后调用，结果如下             |

```bash
python -m scripts.train --data mnist --epochs 3
```
![训练结果](./assets/img01.jpg)
### 问题记录
1. 使用 python scripts/train.py --data mnist --epochs 3默认将scripts当作工作区，将无法获取mycv模块  
解决办法 使用 python -m 则将你的工作区设置为你当前终端所在目录
