#!/bin/bash
set -x

pip install -q scipy tqdm pycocotools gdown
git clone https://github.com/fredzzhang/detr.git

##### Download dataset  and annotation files #####
wget -q --show-progress -O hico_20160224_det.tar.gz "https://www.dropbox.com/scl/fi/tpbrlpcheqa68vy06bhsa/hico_20160224_det.tar.gz?rlkey=xl5i3k3rffm1sup68nekfp0h9&dl=0"
tar -xf hico_20160224_det.tar.gz --no-same-owner
rm hico_20160224_det.tar.gz

# PPDM annotations
wget -q --show-progress -O trainval_hico.json "https://www.dropbox.com/scl/fi/pseygkwdglqo0fjeu3sk2/trainval_hico.json?rlkey=twx3panz8quudyla72tq0yd8x&dl=0"
wget -q --show-progress -O test_hico.json "https://www.dropbox.com/scl/fi/hjcizyd0imfyuto2b0qrw/test_hico.json?rlkey=xi02zpv891339r2fw2xqxc1y0&dl=0"

# PViC annotations
wget -q --show-progress "https://github.com/fredzzhang/hicodet/raw/main/instances_test2015.json"
wget -q --show-progress "https://github.com/fredzzhang/hicodet/raw/main/instances_train2015.json"

##### Download pretrained models #####
wget -q --show-progress https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
gdown --fuzzy "https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing"

##### Move files to directories #####
mkdir checkpoints
mv detr-r50-e632da11.pth checkpoints
mv detr-r50-hicodet.pth checkpoints
mv trainval_hico.json hico_20160224_det
mv test_hico.json hico_20160224_det
mv instances_test2015.json hico_20160224_det
mv instances_train2015.json hico_20160224_det
