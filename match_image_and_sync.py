# -*- coding: utf-8 -*-
"""
Finds ThorSync directories corresponding to ThorImage directories
by looking for filenames or a descriptions of pins used in each
presentation, and matching that to the pins the Arduino says it 
uses, decoded from the ThorSync data.

Need to have pin signalled as in my OlfStimDelivery.ino script and 
follow my practice of copying something containing either the 
filename or the full pins to be presented in to the Notes field
in the ThorImage Capture settings.

Created on Wed Mar 15 15:02:09 2017

@author: Tom O'Connell
"""

import os
import shutil
import xml.etree.ElementTree as etree
import ast
import re
import pickle
import glob
import tom.odors


def get_fly_id(directory_name):
    # may fail if more underscores in directory name

    return '_'.join(d.split('_')[:2])


def get_exptime(thorimage_dir):
    expxml = os.path.join(thorimage_dir, 'Experiment.xml')
    return int(etree.parse(expxml).getroot().find('Date').attrib['uTime'])


def get_synctime(thorsync_dir):
    syncxml = os.path.join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return os.path.getmtime(syncxml)


# warn if has SyncData in name but fails this?
def is_thorsync_dir(d):
    files = {f for f in os.listdir(d)}

    have_settings = False
    have_h5 = False
    for f in files:
        # checking for substring
        if 'ThorRealTimeDataSettings.xml' in f:
            have_settings = True
        if '.h5':
            have_h5 = True

    return have_h5 and have_settings


def is_thorimage_dir(d):
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


# in here should be a directory called 'stimuli'
# which should have pickle files generated by *_odor_randomizer.py
stimuli_base_dir = 'D:/Hong/Tom/stimuli'

thorimage_dir = 'D:/Hong/Tom/flies/tmp_for_unpacking'
thorsync_dir = 'D:/Hong/Tom/flies/tmp_for_unpacking'
sync_dirs = sorted([os.path.join(thorsync_dir,d) for d in os.listdir(thorsync_dir) \
        if is_thorsync_dir(d)], key=get_synctime)

#ordered_sync_dirs = [x for x,y in sorted([(d, get_synctime(d)) \
#        for d in sync_dirs], key=lambda x: x[1])]

valid_fly_ids = re.compile(r'^(\d{6}_\d{2}(?:e|c)?_)')

image_dirs = [os.path.join(thorimage_dir,d) for d in os.listdir(thorimage_dir) \
              if os.path.isdir(os.path.join(thorimage_dir,d)) and valid_fly_ids.match(d)]

'''
print(os.listdir(d))
print(image_dirs)
s1 = set(os.listdir(d))
s2 = set(image_dirs)
print(s1-s2)
'''

# TODO flag to override assertion errors or to ask y/n instead
# TODO add colors to warnings and green if things pass

# doesn't actually check for the ../../stimuli/ prefix
# assumes in is in stimuli_base_dir above
stimfile_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{6}\.p)')

# are angle / square brackets special chars?
#pinlist_re = re.compile(r'\[\{(?:\d\d?(?:, )?)+\}(?:, )?\]+')
pinlist_re = re.compile(r'\[(?:\[(?:\{(?:\d\d?(?:, )?)+\}(?:, )?)+\](?:, )?)+\]')
pin_odor_port_re = re.compile(r'(\d\d?) -> (.* 1e-?\d\d?) -> ([A-Z])')

fly2pl = dict()
fly2ac = dict()
fly2stim = dict()

fly_ids = []

stimfile2imgdir = dict()
without_stimfile = []

for img_dir in image_dirs:
    # TODO does this work?
    fly_id = get_fly_id(os.path.split(img_dir)[-1])
    fly_ids.append(fly_id)
    #img_dir = os.path.join(d,o)

    in_img_dir = [o for o in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir,o))]
    possible_thorsync_dirs = [d for d in in_img_dir if is_thorsync_dir(d)]

    if len(possible_thorsync_dirs) > 1:
        raise Exception('found too many possible thorsync directories in ' + str(o) + \
                '. should only ever be at most one.')

    elif len(possible_thorsync_dirs) == 1:
        print('skipping ' + str(img_dir) + ' because found nested ThorSync directory')
        # skip this directory, because it already seems matched
        continue

    expxml = os.path.join(img_dir, 'Experiment.xml')
    notes = etree.parse(expxml).getroot().find('ExperimentNotes').attrib['text']
    stimfile = stimfile_pattern.search(notes)

    # TODO warn if more than one found?
    
    # TODO unit test
    if stimfile is not None:
        #print('stimfile')
        stimfile = os.path.join(stimuli_base_dir, stimfile.group(0))
        stimfile2imgdir[stimfile] = img_dir
        with open(stimfile, 'rb') as f:
            acs_raw, odor_list_raw = pickle.load(f)
            acs_loaded = tuple(tuple(sorted(l, key=lambda x: x[0])) for l in acs_raw)
            odor2pin_list = [{trip[1]: trip[0] for trip in acs} for acs in acs_loaded]
            # TODO convert odorset list to pinset list
            pl_loaded = tuple(tuple(set(d[o] for o in s) for s in l) \
                              for l, d in zip(odor_list_raw, odor2pin_list))
    else:
        acs_loaded = None
        pl_loaded = None
        
        without_stimfile.append(img_dir)
        
        print(fly_id)
        
    # TODO check loaded stuff is consistent with everything else (including order?),
    # and if it is, add it to same dict. directly comparable? tuple?
        
    pl_match = pinlist_re.search(notes)
    pl = None
    
    # TODO check loaded information with parsed information
    if pl_match is not None:
        if pl_loaded is None:
            print('order')
        pl = ast.literal_eval(pl_match.group(0))
        pl = tuple(tuple(l) for l in pl)
        
        # IF WE HAVE DATA FOR THIS SPECIFIC SESSION, this checks that it is 
        # apparently consistent with information from flies with the same id
        if fly_id in fly2pl:
            assert pl == fly2pl[fly_id], 'apparently conflicting list of pins' + \
                ' presented across flies with the same id: ' + fly_id
        else:
            fly2pl[fly_id] = pl
            
        # checks consistency with pin order loaded from pickle file
        # currently will only work if full pin order can be parsed from
        # notes in each session
        if pl_loaded is not None:
            assert pl == pl_loaded, 'pin order loaded from stimulus pickle ' + \
                'file is apparently inconsistent with pin order in notes. ' + \
                fly_id
    
    #elif pl_loaded is not None:
    if pl_loaded is not None:
        # check might be unnecessary
        if fly_id in fly2pl:
            assert pl_loaded == fly2pl[fly_id], 'apparently conflicting ' + \
                'list of pins presented across flies with ' + \
                'the same id (one of which was loaded from pickle): ' + fly_id
        else:
            fly2pl[fly_id] = pl_loaded
    
    # assumes connections will be printed out in a consistent order
    # across all files compared here (to check equality of parsed tuples)
    # also assumes that if there are multiple sets of connections printed, 
    # they are separated by the word 'stoppers' somewhere
    
    # will only use first 3.
    all_connections = [pin_odor_port_re.findall(n) for n in notes.split('stoppers')[:3]]
    
    all_connections = tuple(tuple((int(x[0]), tom.odors.str2pair(x[1]), x[2]) for x in ac) \
                            for ac in all_connections if len(ac) > 0)
    
    # TODO careful. this may break easily
    if len(all_connections) > 1:
        if acs_loaded is None:
            print('connections')

        # IF WE HAVE DATA FOR THIS SPECIFIC SESSION, this checks that it is 
        # apparently consistent with information from flies with the same id
        if fly_id in fly2ac:
            assert all_connections == fly2ac[fly_id], 'apparently conflicting ' + \
                'pin (valve) to odor connections presented across flies with ' + \
                'the same id: ' + fly_id
        else:
            fly2ac[fly_id] = all_connections
            
        if acs_loaded is not None:
            assert all_connections == acs_loaded, 'connections described in notes ' + \
                'are apparently inconsistent with those in stimulus pickle file ' + \
                fly_id + '\n\nnotes:' + str(all_connections) + '\n\nloaded:' + str(acs_loaded)
                
    if acs_loaded is not None:
        # check might be unnecessary
        if fly_id in fly2ac:
            assert acs_loaded == fly2ac[fly_id], 'apparently conflicting ' + \
                'pin (valve) to odor connections presented across flies with ' + \
                'the same id (one of which was loaded from pickle): ' + fly_id
        else:
            fly2ac[fly_id] = acs_loaded

fly_ids = set(fly_ids)
print('Processing fly IDs:')
print(fly_ids)

# TODO warn and include list of things we dont have stimulus for at this point
# mappings and pins separately?
    
# could happen by chance with reasonable probability if you have a really
# small odor panel

# TODO write unit tests for these!!!

# TODO but this will happen if something is run twice
# maybe just delete in advance trials that went badly?
# checks the same pin order isn't observed twice in recording
# sessions with different fly_ids (even though not all recordings
# list this)
# would likely indicate a data entry error
for fly in fly2pl:
    copy = dict(fly2pl)
    del copy[fly]
    assert fly2pl.values() != copy.values(), 'possible duplication data entry ' + \
        'error (' + fly + ' and at least one other are involved)'

# checks the same connection pattern isn't observed twice in recording
# sessions with different fly_ids. same reason.
for fly in fly2ac:
    copy = dict(fly2ac)
    del copy[fly]
    assert set(fly2ac.values()) != set(copy.values()), \
        'possible duplication data entry error (' + fly + ' and at least one other are involved)'

assert len(stimfile2imgdir.keys()) == len(stimfile2imgdir.values()), \
    'this should not happen, but dont want to do something dangerous'

# if we have not failed any checks so far, move the stimulus files 
# to their matching ThorImage directories
for sf in stimfile2imgdir:
    dst = stimfile2imgdir[sf]
    assert os.path.isdir(dst)
    # should copy (some?) operating system metadata too
    shutil.copy2(sf, dst)

# generate correct pin 2 odor mappings from notes in available
# save in ThorImage directory as differently named pickle file
# to be read by analysis software
print('Valve to odor connections and stimulus order pickle file not found for:')
for d in without_stimfile:
    print(d)
print('')

fly2ordered_img_dirs = dict()
for fid in fly_ids:
    # TODO sorted in correct order?
    fly2ordered_img_dirs[fid] = sorted([d for d in image_dirs if fid in d], key=get_exptime)
    #fly2ordered_img_dirs[fid] = [d for d, t in sorted([(d, get_exptime(d)) \
    #        for d in img_dirs if fid in d], key=lambda x: x[1])]
    fly2ordered_img_dirs[fid] = dict(zip(fly2ordered_img_dirs[fid], \
            range(len(fly2ordered_img_dirs[fid]))))

# TODO d enough or need to join?
for d in without_stimfile:
    # TODO handle this as well later
    '''
    if get_fly_id(d) in fly2pl:
        odor_list =
    '''

    # TODO need to split d?
    fly_id = get_fly_id(d)

    if fly_id in fly2ac:
        # get the index of the list of different connections (# elements = # recording sessions)
        # so we can save only the relevant connections in the directory for that
        # recording session
        connections = None
        for i, session in enumerate(fly2ac[fly_id]):
            if i == fly2ordered_img_dirs[fly_id][d]:
                connections = list(session)
        print('connections', connections)
        print('saving valve to odor connections from notes, for ' + fly_id + ', to ' + d)

        gen_connections = os.path.join(d, 'generated_pin2odor.p')
        with open(gen_connections, 'wb') as f:
            pickle.dump(connections, f)

# for each pickle file, find the ThorSync file that matches that 
# pin presentation order after decoding

# TODO make sure no two SyncData files decode to the same thing
# ALTHOUGH THIS COULD HAPPEN... MAYBE DONT
# TODO if >1 SyncData decode to same thing (and loaded parsed data does too)
# match them up in time and warn
# move matching to imaging directories

# if any imaging directories do not appear to have corresponding
# ThorSync data, there is a serious problem that the user needs
# to address

# move non-matching to orphans

