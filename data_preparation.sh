#!/bin/bash

cd ARKitScenes_processed
cat train.tar.part_0{00,01,02} > train.tar
md5sum -c MD5SUMS.txt
cd ..
mkdir ./data
mkdir ./data/arkit
mv ARKitScenes_processed/* ./data/arkit/
cd ./data/arkit/
tar xvf train_points.tar ./
tar xvf val_points.tar ./
tar xvf val.tar ./
mkdir points
mv train_points/* ./points/
mv val_points/* ./points/
mkdir processed
mkdir processed/3dod/
mv train ./processed/3dod/
mv val ./processed/3dod/
echo "Data preparation completed!"
