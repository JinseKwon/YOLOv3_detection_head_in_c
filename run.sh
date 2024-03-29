#!/bin/bash

g++ yolov3.c yolov3_det_head.c -o yolov3 -I. -I../onnxruntime-linux-x64-1.11.0/include/ -fPIC ../onnxruntime-linux-x64-1.11.0/lib/libonnxruntime.so -lm `pkg-config --libs opencv`

if [ $# -eq 0 ] ; then
	./yolov3 yolov3_re.onnx dog.jpg cpu
else
	./yolov3 yolov3_re.onnx $1 cpu
fi
