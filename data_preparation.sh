#!/bin/bash

cd ScanNet_processed
cat posed_images.tar.part_0{00,01,02} > posed_images.tar
md5sum -c MD5SUMS.txt
cd ..
mkdir ./data
mkdir ./data/scannet
mv ScanNet_processed/* ./data/scannet/
cd ./data/scannet/
tar xvf posed_images.tar ./
tar xvf points.tar ./
echo "Data preparation completed!"
