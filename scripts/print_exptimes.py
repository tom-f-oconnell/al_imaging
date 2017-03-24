#!/usr/bin/env python3

import os
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

    if have_xml and tifs > 1:
        return True
    else:
        return False

image_dirs = [d for d in os.listdir() if is_thorimage_dir(d)]
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
