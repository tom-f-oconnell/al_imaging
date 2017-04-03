#!/usr/bin/env python3

import os
from os.path import join
import xml.etree.ElementTree as etree

from al_imaging import util

exp_dir = util.get_expdir()
if not exp_dir:
    exp_dir = '.'

image_dirs = [join(exp_dir,d) for d in os.listdir(exp_dir) if util.is_thorimage_dir(join(exp_dir,d))]
out = sorted([(util.get_readable_exptime(d), d) for d in image_dirs], key=lambda x: x[0])

last_day = None
for e in out:
    # hack to fix platform (?) dependence of datetime string
    if len(e[0].split('/')) == 1:
        day = e[0].split('-')[1]
    else:
        day = e[0].split('/')[1]
        
    if last_day is not None and last_day != day:
        print('')
    print(e[0], e[1])
    last_day = day
