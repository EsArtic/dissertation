#!/usr/bin/env sh

rm -r ./classification/*
rm -r ./localization/*

if [ ! -d "./classification/test" ];
then
    mkdir ./classification/test
fi

if [ ! -d "./classification/train" ];
then
    mkdir ./classification/train
fi

if [ ! -d "./localization/test" ];
then
    mkdir ./localization/test
fi

if [ ! -d "./localization/train" ];
then
    mkdir ./localization/train
fi

echo "Extracting data of test set..."
tar -xf VOCtest_06-Nov-2007.tar
cp -r ./VOCdevkit/VOC2007/JPEGImages/* ./classification/test/
python ../scripts/extract_data.py -p ./VOCdevkit/VOC2007/JPEGImages -a ./VOCdevkit/VOC2007/Annotations -t
rm -r ./VOCdevkit

echo "Extracting data of train set..."
tar -xf VOCtrainval_11-May-2012.tar
python ../scripts/extract_data.py -p ./VOCdevkit/VOC2012/JPEGImages -a ./VOCdevkit/VOC2012/Annotations
rm -r ./VOCdevkit

