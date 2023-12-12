pip install gdown
gdown https://drive.google.com/uc?id=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS
gdown https://drive.google.com/uc?id=e1i5tkMYDGPtJ6oajQ9E4XoDAn7Dn2ERLN
mkdir coco
mv instances_train2017.json coco
tar -xvf hico_20160224_det.tar.gz --no-same-owner
python3 hico_det_convert.py
python3 extract_coco_classes.py --contiguous