#!/bin/bash
for filename in ../../test/simulation/math/*.mo; do
    ./marco --omc-bypass --end-time=1 -o simulation $filename
    ./simulation | filecheck $filename
done