#!/bin/bash

for d in */; do ls $d*ChanB* | (if [ $(wc -l) -ne 0 ]; then echo $d; fi); done 2> /dev/null
