#!/bin/bash

g++ yolov3_det_head.c -o yolov3_det_head -lm `pkg-config --libs opencv`

./yolov3_det_head
