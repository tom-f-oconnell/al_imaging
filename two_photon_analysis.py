# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

import importlib
import tom.analysis
importlib.reload(tom.analysis)

# TODO this syntax is frowned upon. just alias analysis.
from tom.analysis import *
import argparse
import glob

import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

# TODO make this repository
parser = argparse.ArgumentParser(description='Analysis pipeline for 2p movies of ' +\
        'GCaMP signals in the antennal lobe after odor presentation.',
        epilog='See github.com/tom-f-oconnell/al-imaging for more information.')

show_parser = parser.add_mutually_exclusive_group(required=False)
show_parser.add_argument('-s', '--showplots', dest='show_plots', action='store_true', \
        help='show plots at end')
show_parser.add_argument('-ns','--noshowplots', dest='show_plots', action='store_false', \
        help='(default) do not show plots at end')

print_summary_parser = parser.add_mutually_exclusive_group(required=False)
print_summary_parser.add_argument('-p', '--printsummary', dest='print_summary_only', \
        action='store_true', help='output summary of experiments, including which odor was ' + \
        'presented during each frame range in each input TIF.')
print_summary_parser.add_argument('-a','--analysis', dest='print_summary_only', \
        action='store_false', help='(default) run analysis')

parser.add_argument('-t', '--test', dest='test', action='store_true')
parser.set_defaults(show_plots=False, print_summary_only=False, test=False)

# automatically parses from sys.argv when called without arguments
args = parser.parse_args()

# TODO close one plot before next is saved if we aren't going to display at the end?

# initialize seaborn, to make its tweaks to matplotlib (should make things look nicer)
# TODO can't figure out how to actually change default linewidth
#sns.set_style('darkgrid')
# TODO test. does this mean green is plotted first? (it would seem to)
#sns.set_palette('GnBu_d')

#prefix = '/media/threeA/hong/flies/tifs/xy_motion_corrected/'
prefix = '/media/threeA/hong/flies/tifs/thunder_registered/'

# TODO need to expand to include _o1/2/3/anat suffixes
# means I will need to fix broken file names

if args.test:
    flies = {'Mock': 
              ('170212_01c',)}

    pin2odor_names = {'Mock':
                       ('2017-02-12_163940.p',)}

else:
    flies = {'Mock': 
              ('170212_01c',
               '170213_01c',
               '170213_02c'),
             '2-butanone 1e-4': 
              ('170214_01e',
               '170215_01e',
               '170215_02e')}
#)

    # TODO just put these in a consistent position relative to other files ->
    # maybe assert date / time is close enough to when other files were generated
    pin2odor_names = {'Mock':
                       ('2017-02-12_163940.p',
                        '2017-02-13_145233.p',
                        '2017-02-13_174625.p'),
                      '2-butanone 1e-4':
                       ('2017-02-14_145733.p',
                        '2017-02-15_095301.p',
                        '2017-02-15_135620.p')}

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
            # odor_connections are saved as triples of (arduino pin / valve, odor, manifold port)
            for odor_connection in experiment:
                odor_panel.add(odor_connection[1])

# TODO get corresponding data
#'2017-02-15_135620.p')

suffix = '.tif'

# TODO i feel kinda like i should have all loading of files in this script and 
# only deal with raw data in tom/analysis
# TODO TODO TODO refactor. make more neat to switch between printing info and running analysis.
# though the goal was shorter code and time-to-plots of future experiments. helping?

secs_before = 3
secs_after = 12
trial_duration = secs_before + secs_after

# mock / butanone reared
for condition in sorted(files.keys()):
    for full, nick, all_p2o  in zip(files[condition], flies[condition], pin2odors[condition]):
        # for each odor panel (<=3 per fly, not all with same # of vials)

        p2o_dicts = []
        syncdata_files = []
        imaging_files = []

        for o, p2o in zip(('_o1', '_o2', '_o3'), all_p2o):

            if not os.path.exists(full + o + suffix):
                print('could not find ' + full + o + suffix + '. skipping')
                continue

            # assumes p2o is an iterable of triplets, of format (pin, odor, manifold_port)
            # TODO override in special cases where note indicates further exception
            if o == '_o3':
                # TODO warn!
                # TODO check against notes for cases using this and make sure ordering
                # of odors is correct (s.t. when zipped with pins mapping is correct)
                # first 3 pins, 5,6,and 7
                p2o_dict = dict(zip(range(5,8), map(lambda x: x[1], p2o)))
            else:
                p2o_dict = dict(map(lambda x: (x[0], x[1]), p2o))

            thorsync_file = glob.glob('/media/threeA/hong/flies/' \
                                + nick + o + '/SyncData*')[0] + '/Episode001.h5'

            imaging_file = full + o + suffix

            if args.print_summary_only:
                print_odor_order(thorsync_file, p2o_dict, imaging_file, trial_duration)

            p2o_dicts.append(p2o_dict)
            syncdata_files.append(thorsync_file)
            imaging_files.append(imaging_file)

        if not args.print_summary_only:
            fly_df = process_2p(imaging_files, syncdata_files, secs_before=3, secs_after=12, \
                    pin2odor=p2o_dicts)

            '''
            g = sns.FacetGrid(fly_df.reset_index(), hue='trial', col='odor', col_wrap=5)
            # TODO
            g = g.map(plt.plot, 'frame', glom)
            '''

            # TODO merge fly_dfs across flies. could probably also just use append.

            #fly_df.isnull().values.ravel().sum()

            # summarizes responses from each glomerulus for each individual fly
            # TODO nested subplots?
            glomeruli = set(fly_df.columns)
            print(glomeruli)
            glomeruli.remove('block')
            print(glomeruli)

            # TODO just make sublots each block?
            for glom in glomeruli:

                print(glom)
                # TODO assert somehow that a block either has the glomerulus in all frames / odors
                # or doesnt?
                # get the entries (?) that have data for that glomerulus
                glom_df = fly_df[glom][pd.notnull(fly_df[glom])]
                #containing_blocks = fly_df['block'][pd.notnull(fly_df[glom])].reset_index()['block']
                containing_blocks = fly_df[pd.notnull(fly_df[glom])].reset_index()['block']

                # TODO change title of y axis to dF/F (might require changing col name)
                # and add title w/ glom name
                # TODO grid? units of seconds on x-axis. patch in stimulus presentation.
                # TODO check for onsets before we would expect them
                df = glom_df.reset_index()
                g = sns.FacetGrid(df, hue='trial', \
                        col='odor', col_wrap=5)
                g = g.map(plt.plot, 'frame', glom)
                g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
                title = glom + ' from blocks ' + str(containing_blocks.unique())
                g.fig.suptitle(title)
                print(df.columns)
                # get set of blocks occurring with current odor (column name, the arg to lambda)

                f = lambda x: x + ' ' + \
                        str(list(filter(lambda e: type(e) is int, set(containing_blocks.where(df['odor'] == x).unique()))))

                g.set_titles(col_func=f)
                #for t in

            # TODO mean plot w/ errorbars



# TODO for each odor known to be a private odor (do i have all the glomeruli i'm interested in
# covered here?), see which region lights up (dis/con-junction?)
# (can restrict to only odors coverd in intersection of values sets of pin2odor mappings)
# and use those regions as definitions of glomeruli to analyze responses to public odors

# TODO plot sums in this regions as traces over time

# TODO compare those regions to where you'd expect them to be from anatomical information alone
# or do at least their relative positions make sense?

if args.show_plots:
    plt.show()
