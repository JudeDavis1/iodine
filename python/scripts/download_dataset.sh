echo "Downloading FreiHAND dataset..."

wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip -v
unzip FreiHAND_pub_v2_eval.zip -d data/

echo "Cleaning up..."

rm FreiHAND_pub_v2_eval.zip
rm -rf data/evaluation/anno
rm -rf data/evaluation/colormap
rm -rf data/evaluation/facemap
rm -rf data/evaluation/segmap
rm -rf data/evaluation/vert_offset_map

rm data/evaluation_scale.json
rm data/evaluation_mano.json
rm data/evaluation_K.json
rm data/evaluation_errors.json

mv data/evaluation_xyz.json data/freihand_train.json
