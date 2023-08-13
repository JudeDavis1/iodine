echo "Downloading FreiHAND dataset..."

wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -v
unzip FreiHAND_pub_v2.zip -d data/

wget https://download.openmmlab.com/mmpose/datasets/frei_annotations.tar
tar -xvf frei_annotations.tar
rm frei_annotations.tar

mv annotations/freihand_train.json data/
rm -rf annotations

cd data/training/rgb && python3 -c "
import os

for filename in os.listdir('.'):
    if filename.startswith('0') and filename.endswith('.jpg'):
        new_filename = filename.lstrip('0')
        os.rename(filename, new_filename)
        print(f'Renamed {filename} to {new_filename}')

"

cd ../../../
mv data/training/rgb/.jpg data/training/rgb/0.jpg


echo "Cleaning up..."

rm FreiHAND_pub_v2.zip
rm -rf data/evaluation
rm -rf data/training/mask
rm data/training_*
