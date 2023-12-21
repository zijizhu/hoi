pip install scipy
pip install tqdm
pip install pycocotools
pip install torchmetrics
git clone https://github.com/fredzzhang/detr.git
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
wget -O instances_train2017.json "https://www.dropbox.com/scl/fi/k6p0aeajzzab935bxgq2v/instances_train2017.json?rlkey=qvqmn526dtu2hylcc8imxtc7s&dl=0"
wget -O hico_20160224_det.tar.gz "https://www.dropbox.com/scl/fi/tpbrlpcheqa68vy06bhsa/hico_20160224_det.tar.gz?rlkey=xl5i3k3rffm1sup68nekfp0h9&dl=0"
mkdir coco
mkdir checkpoints
mv instances_train2017.json coco
mv detr-r50-e632da11.pth checkpoints
tar -xvf hico_20160224_det.tar.gz --no-same-owner
python3 hico_det_convert.py
python3 extract_coco_classes.py --contiguous
