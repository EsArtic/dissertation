#!/usr/bin/env sh

./caffe/build/tools/caffe train \
    --solver=prototxt/localization_GoogleNet/solver.prototxt --weights=caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel 2>&1 | tee ./log/$(date +%s)_localization_train.log

# ./caffe/build/tools/caffe train \
#     --solver=prototxt/localization_AlexNet/solver.prototxt --weights=caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel 2>&1 | tee ./log/$(date +%s)_localization_train.log
