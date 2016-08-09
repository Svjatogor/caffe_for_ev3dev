#!/bin/bash
set -e

## your path to caffe sources
#CAFFE_ROOT='/Users/nik/Developer/Components/CodeGen/caffe/'
: ${CAFFE_ROOT:?"Set CAFFE_ROOT to caffe sources"}

TARGET_NAME='CaffeScoring'

RELATIVE_TO_RUN_PLACE="`dirname \"$0\"`"
SCRIPT_LOCATION="`( cd \"${RELATIVE_TO_RUN_PLACE}\" && pwd )`"
DATA_LOCATION=${SCRIPT_LOCATION}/data
GIT_ROOT=${SCRIPT_LOCATION}/../

rm -rf ${DATA_LOCATION}/ilsvrc12_val_lmdb
echo "---------->Convert images to lmdb format<-----------"
${CAFFE_ROOT}/build/tools/convert_imageset \
--resize_height=256 \
--resize_width=256 \
${DATA_LOCATION}/images/ \
${DATA_LOCATION}/val.txt \
${DATA_LOCATION}/ilsvrc12_val_lmdb

rm -rf ${DATA_LOCATION}/bvlc_googlenet.caffemodel
echo "---------->Downloading caffe model of pretrained GoogLeNet<-----------"
${CAFFE_ROOT}/scripts/download_model_binary.py ${CAFFE_ROOT}/models/bvlc_googlenet
cp -v ${CAFFE_ROOT}/models/bvlc_googlenet/bvlc_googlenet.caffemodel ${DATA_LOCATION}

echo "---------->Build ${TARGET_NAME}<-----------"
cd ${GIT_ROOT}; mkdir -p build; cd build; cmake ../; make clean; make ${TARGET_NAME} -j8;

export LD_LIBRARY_PATH=${CAFFE_ROOT}/build/lib
echo "---------->Launching ${TARGET_NAME}<-----------"
./${TARGET_NAME}/${TARGET_NAME} \
${SCRIPT_LOCATION}/data/test_val.prototxt \
${SCRIPT_LOCATION}/data/bvlc_googlenet.caffemodel \
${SCRIPT_LOCATION}/data/synset_words.txt \
3