set -x

pip install scipy tqdm pycocotools
git clone https://github.com/fredzzhang/detr.git

##### Download dataset  and annotation files #####
wget -q --show-progress -O hico_20160224_det.tar.gz "https://www.dropbox.com/scl/fi/tpbrlpcheqa68vy06bhsa/hico_20160224_det.tar.gz?rlkey=xl5i3k3rffm1sup68nekfp0h9&dl=0"
tar -xf hico_20160224_det.tar.gz --no-same-owner

wget -q --show-progress -O trainval_hico.json "https://www.dropbox.com/scl/fi/pseygkwdglqo0fjeu3sk2/trainval_hico.json?rlkey=twx3panz8quudyla72tq0yd8x&dl=0"
wget -q --show-progress -O test_hico.json "https://www.dropbox.com/scl/fi/hjcizyd0imfyuto2b0qrw/test_hico.json?rlkey=xi02zpv891339r2fw2xqxc1y0&dl=0"

##### Download pretrained model #####
wget -q --show-progress https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

mkdir checkpoints
mv detr-r50-e632da11.pth checkpoints
mv trainval_hico.json hico_20160224_det
mv test_hico.json hico_20160224_det
