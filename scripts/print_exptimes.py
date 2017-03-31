#!/usr/bin/env python3

import os
from os.path import join
import xml.etree.ElementTree as etree

def get_readable_exptime(thorimage_dir):
    expxml = os.path.join(thorimage_dir, 'Experiment.xml')
    return etree.parse(expxml).getroot().find('Date').attrib['date']

def is_thorimage_dir(d):
    if not os.path.isdir(d):
        return False
    
    files = {f for f in os.listdir(d)}

    have_xml = False
    tifs = 0
    for f in files:
        if 'Experiment.xml' in f:
            have_xml = True
        elif '.tif' in f:
            tifs += 1

    if have_xml and tifs >= 1:
        return True
    else:
        return False

expdir_envvar = 'IMAGING_EXP_DIR'
if expdir_envvar in os.environ:
    exp_dir = os.environ[expdir_envvar]
else:
    exp_dir = '.'

image_dirs = [join(exp_dir,d) for d in os.listdir(exp_dir) if is_thorimage_dir(join(exp_dir,d))]
out = sorted([(get_readable_exptime(d), d) for d in image_dirs], key=lambda x: x[0])

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
