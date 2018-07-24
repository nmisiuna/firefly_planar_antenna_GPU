#!/bin/sh

rm firefly.o
rm link.o
rm nicklib_GPU.o
rm firefly_pos.o
nvcc -ccbin=gcc -arch=sm_20 -dc firefly.cu ../../../nicklib/nicklib_GPU.cu firefly_pos.cu -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
nvcc -ccbin=gcc -arch=sm_20 -dlink firefly.o nicklib_GPU.o firefly_pos.o --output-file link.o