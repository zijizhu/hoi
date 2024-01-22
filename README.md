# Experimental Code for Human Object Interaction Detection

## Run Experiments

Evaluation with detr-resnet50:

```bash
python main.py --device cuda --eval --pretrained checkpoints/detr-r50-e632da11.pth
```

Evaluation with detr-resnet50 finetuned on hicodet:

```bash
python main.py --eval --device cuda --pretrained checkpoints/detr-r50-e632da11.pth --resume checkpoints/detr-r50-hicodet.pth
```