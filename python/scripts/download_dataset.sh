echo "Downloading FreiHAND dataset..."

wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -v
unzip FreiHAND_pub_v2.zip -d data/

echo "Cleaning up..."

rm FreiHAND_pub_v2.zip
rm -rf data/evaluation
rm -rf data/training/mask

mv data/training_verts.json data/freihand_train.json
