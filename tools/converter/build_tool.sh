#!/bin/bash

if [ -d "build" ]; then
  rm -rf build
fi
./generate_schema.sh

mkdir build
cd build

cmake .. -DCMAKE_PREFIX_PATH=/home/guo/protobuf3-install/
make clean
make -j16
