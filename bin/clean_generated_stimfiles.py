#!/usr/bin/env python3

import re
import os
import hashlib

from al_imaging import util

# TODO get relative to expdir?
stimuli_base_dir = 'D:/Hong/Tom/stimuli'

# from answer on StackOverflow by quantumSoup
def md5(fname):
    m = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


# TODO factor either of these into util?
def matching_stimfile(fname):
    return os.path.join(stimuli_base_dir, os.path.split(f)[-1])


def stim_matches(fname):
    exists = os.path.isfile(matching_stimfile(fname))
    hash_matches = md5(fname) == md5(matching_stimfile(fname))
    
    # TODO os module equivalence?
    assert fname != matching_stimfile(fname), 'comparing hash ' + \
        'meaningless if files are the same'
    
    if not (exists and hash_matches):
        print(fname + ' could not be deleted. parent file exists?', exists, \
              'hash matches?', hash_matches)
        return False
    
    else:
        return True


thorimage_dir = util.get_expdir()
if not thorimage_dir:
    thorimage_dir = 'D:/Hong/Tom/flies'

image_dirs = [os.path.join(thorimage_dir,d) for d in os.listdir(thorimage_dir)]
image_dirs =  [d for d in image_dirs \
        if os.path.isdir(d) and util.valid_fly_ids.search(d)]

deleted_something = False

for d in image_dirs:
    #print(d)
    for f in [os.path.join(d,f) for f in os.listdir(d)]:
        if '.p' in f and os.path.isfile(f):
            if 'generated' in f or stim_matches(f):
                #print('would remove ' + f)
                os.remove(f)
                deleted_something = True
                
if not deleted_something:
    print('no pickle files to clean.')
