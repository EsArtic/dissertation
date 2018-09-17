#!/usr/bin/env sh

cd scripts
for i in `seq 1 6`
do
   python predict.py -s ../example/images/$i.jpg -c ../models/classification_GoogleNet.caffemodel \
       -l ../models/localization_GoogleNet.caffemodel -d ../example/predicts
done
