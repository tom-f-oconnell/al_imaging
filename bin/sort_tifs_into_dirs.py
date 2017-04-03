#!/usr/bin/env python3

from os import listdir, remove, rename
from os.path import isfile, isdir, split, join, exists

from al_imaging import util

exp_dir = util.get_expdir()
if not exp_dir:
    exp_dir = '/home/tom/data/flies'

make_changes = True
tif_dir = '/home/tom/data/flies/autotifs'

files = [join(tif_dir,f) for f in listdir(tif_dir) if isfile(join(tif_dir,f))]
imaging_dirs = [join(exp_dir, d) for d in listdir(exp_dir) if isdir(join(exp_dir, d))]

for f in files:
    if '_Chan' in f:
        dir_name = split(f)[-1].split('_Chan')[0]

        for d in imaging_dirs:
            if split(d)[-1] == dir_name:
                new = join(d, split(f)[-1])

                if exists(new):
                    print('DELETING', new)
                    if make_changes:
                        remove(new)

                print('moving', f, 'to', new)
                if make_changes:
                    rename(f, new)
                break
    
