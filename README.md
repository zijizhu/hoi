# Experimental Code for Human Object Interaction Detection

## Setup Repository

Clone the repository and download relevant data:
```bash
git clone https://github.com/zijizhu/hoi.git
cd hoi
git submodule update --init
bash setup.sh
```

## Run Experiments

Evaluation with `detr-resnet50`:

```bash
python main.py --device cuda --eval --pretrained checkpoints/detr-r50-e632da11.pth
```

Evaluation with `detr-resnet50` finetuned on `hicodet` (provided [here](https://github.com/fredzzhang/hicodet/tree/main/detections)):

```bash
python main.py --eval --device cuda --pretrained checkpoints/detr-r50-e632da11.pth --resume checkpoints/detr-r50-hicodet.pth
```