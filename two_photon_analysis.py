# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

# imports numpy, pyplot, os, pickle, seaborn
import importlib

import tom.analysis
importlib.reload(tom.analysis)

from tom.analysis import *
import glob


# opencv?

plt.close('all')

# initialize seaborn, to make its tweaks to matplotlib (should make things look nicer)
# TODO can't figure out how to actually change default linewidth
sns.set_style('darkgrid')
# TODO test. does this mean green is plotted first? (it would seem to)
sns.set_palette('GnBu_d')

prefix = '/media/threeA/hong/flies/tifs/xy_motion_corrected/'

# TODO need to expand to include _o1/2/3/anat suffixes
# means I will need to fix broken file names
flies = {'Mock': 
          ('170212_01c',
           '170213_01c',
           '170213_02c'),
         '2-butanone 1e-4': 
          ('170214_01e',
           '170215_01e',
           '170215_02e')}
#)

# TODO wasn't there a second fly on the 15th?
# do any of the earlier flies also have anatomical stacks that are just improperly labeled?
# look at listing of image objects dimensions?

files = dict()
for k in flies:
    files[k] = fix_names(prefix, flies[k], '')

# TODO want plots by glomerulus i think
# for all the glomeruli that i can compare across the two conditions

# TODO any relevant metadata to correspond sync and imaging data? maybe in the referenced DB?
# i couldnt readily see it elsewhere

p2o_prefix = '/media/threeA/hong/pins2odors/'
pin2odor_names = {'Mock':
                   ('2017-02-12_163940.p',
                    '2017-02-13_145233.p',
                    '2017-02-13_174625.p'),
                  '2-butanone 1e-4':
                   ('2017-02-14_145733.p',
                    '2017-02-15_095301.p',
                    '2017-02-15_135620.p')}

pin2odors = dict()

for k in flies:
    # load from pin2odor pickled dicts
    for f in fix_names(p2o_prefix, pin2odor_names[k], ''):
        with open(f, 'rb') as fp:
            if not k in pin2odors:
                pin2odors[k] = []

            pin2odors[k].append(pickle.load(fp))

#pin2odors = [pickle.load(open(f)) for f in fix_names(p2o_prefix, pin2odor_names, '')]
odor_panel = set()
for group in pin2odors.values():
    for fly in group:
        for experiment in fly:
            for odor_connection in experiment:
                odor_panel.add(odor_connection[1])

# TODO get corresponding data
#'2017-02-15_135620.p')

# TODO first, just calc dF/F for each odor and for each fly, and plot those
# also avg for each group
# overlay on (avg? exemplar?) images? what kind of thresholds?

suffix = '_stackregd.tif'

# mock / butanone reared
for condition in files:
    for full, nick, all_p2o  in zip(files[condition], flies[condition], pin2odors[condition]):
        # for each odor panel (<=3 per fly, not all with same # of vials)
        for o, p2o in zip(('_o1', '_o2', '_o3'), all_p2o):

            if not os.path.exists(full + o + suffix):
                print('could not find ' + full + o + suffix + '. skipping')
                continue

            # assumes p2o is an iterable of triplets, of format (pin, odor, manifold_port)
            # TODO override in special cases where note indicates further exception
            if o == '_o3':
                # TODO print when using this
                # first 3 pins, 5,6,and 7
                p2o_dict = dict(zip(range(5,8), map(lambda x: x[1], p2o)))
            else:
                p2o_dict = dict(map(lambda x: (x[0], x[1]), p2o))

            syncdata = glob.glob('/media/threeA/hong/flies/' + nick + o + '/SyncData*')[0] + \
                    '/Episode001.h5'
            process_2p(full + o + suffix, syncdata, secs_after=6, pin2odor=p2o_dict)

# TODO for each odor known to be a private odor (do i have all the glomeruli i'm interested in
# covered here?), see which region lights up (dis/con-junction?)
# (can restrict to only odors coverd in intersection of values sets of pin2odor mappings)
# and use those regions as definitions of glomeruli to analyze responses to public odors

# TODO plot sums in this regions as traces over time

# TODO compare those regions to where you'd expect them to be from anatomical information alone
# or do at least their relative positions make sense?

plt.show()
