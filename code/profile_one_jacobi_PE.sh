#!/bin/bash

if [ $PMI_RANK -eq 0 ]; then
    ncu -s 20 -c 1 -k jacobi -o report -f $@
else
    $@
fi
