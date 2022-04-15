#!/bin/bash

g++ yolov3.c yolov3_det_head.c -o yolov3 -I. -lonnxruntime -lm `pkg-config --libs opencv`

if [ $# -eq 0 ] ; then
	./yolov3 yolov3.onnx dog.jpg cpu
else
	./yolov3 yolov3.onnx $1 cpu
fi
