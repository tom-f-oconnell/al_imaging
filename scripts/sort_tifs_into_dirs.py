#!/usr/bin/env python3

import os
from os.path import isfile, isdir, split, join

tif_dir = '/home/tom/data/autotifs'

exp_envvar = 'IMAGING_EXP_DIR'
if exp_envvar in os.environ:
    exp_dir = os.environ[exp_envvar]

else:
    exp_dir = '/home/tom/data/flies'

files = [join(tif_dir,f) for f in os.listdir(tif_dir) if isfile(join(tif_dir,f))]
imaging_dirs = [join(exp_dir, d) for d in os.listdir(exp_dir) if isdir(join(exp_dir, d))]

for f in files:
    if '_Chan' in f:
        dir_name = split(f)[-1].split('_Chan')[0]

        for d in imaging_dirs:
            if split(d)[-1] == dir_name:
                new = join(d, split(f)[-1])
                print('moving', f, 'to', new)
                os.rename(f, new)
                break
    
