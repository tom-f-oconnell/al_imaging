# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

'''
import importlib
import tom.analysis
importlib.reload(tom.analysis)
'''

# TODO this syntax is frowned upon. just alias analysis.
#from tom.analysis import *
import tom.analysis as ta
import argparse
import glob

import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

# TODO make this repository
parser = argparse.ArgumentParser(description='Analysis pipeline for 2p movies of ' +\
        'GCaMP signals in the antennal lobe after odor presentation.',
        epilog='See github.com/tom-f-oconnell/al_imaging for more information.')

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

#sns.set_style('darkgrid')
#sns.set_palette('GnBu_d')

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

#ta.group_directories(working_directory, ['c', 'e'])

"""
directories -> skip malformed data -> registered stacks in same dirs
-necessary to automatically screen for trials with drift / loss of signal? (do ~here if so)

registered stacks -> smooth+dF/F
                     need info about stimulus, possibly including trigger, to group stack into trials

                    -> average across trials within fly -> reject under threshold
                                                                        for broad odor activation
                                    -> images via some projection
                                       (as directory -> (odor -> image))
                                       -will then have to have a plotting function that deals 
                                        with this, which would probably be less general
                                       -or another function that processes this dict
                                        into one by (fly_id -> (glom -> (odor -> image)))

                                       -> ROI (as directory -> (glom -> OpenCV roi))
                                          (not all images used to make ROIs)
                                       -> burn in scale bar from metadata (optionally)

                        "" +ROIs -> reject bad ROIs (make GUI to speed this up? / prompt user)
                                    + save which ROIs were rejected, recompute if code changes

                                    -> individual traces (in data frame) binned in ROIs
                                         in: ROIs, stacks from same directory (maybe ranges
                                             of frames from within session at a time?), stimuli
                                             corresponding to the whole session or ranges of 
                                             frames currently being processed
                                         out:
                                           (multiindexed dataframe, w/ first index as fly_id 
                                           (derived from directory), and sessions (actual 
                                            directories) as further index)
                                       -maybe add fly_id and session # upon return? or pass in?
                                        in general sessions should be able to have multiple ROIs,
                                        though in my latest experiments, i am aiming for one glom
                                        per session

dataframe -> group by rearing condition, glomerulus, and stimuli (???)

plots i want:
    -fly summaries:
        -grid for each glomerulus, w/ max projection of avg for each odor
        -ROIs for each glomerulus (private odors now only exist in sessions w/ suffix of that
         glomerulus)
        -individual traces
        -mean + confidence intervals
    -averages across flies for each, separate for mock and 2-butanone reared 
     (glomeruli, stimulus) mean + confidence intervals
        -one grid per each glomerulus (row w/ private, another row w/ private + PA)
        -(?) fit slope / sigmoid & summarize w/ 1 number per glomerulus, for private and
         private + PA
            -include mock on private series, and include PA on priv + PA row
"""

"""
plotting functions i want:
    -make a grid of images for each terminal dictionary, and title the grids with
     keys of any containing dictionaries
        -can use for ROIs and projections
    -wrapper around facetgrid stuff?
       -sem or bootstrapping
       -individual traces
"""

secs_before = 3
secs_after = 12
trial_duration = secs_before + secs_after

experiment_directory = '/media/threeA/Tom/flies'

substring2condition = {'c': 'mock reared',
                       'e': '2-butanone 1e-4 reared'}

# variables in the Arduino code. other parameters will be loaded from
# ThorImage and ThorSync XML metadata, or pickle files describing stimuli
stim_params = {'ITI_s': 30,
               'odor_pulse_ms': 500,
               'repeats': 5,
               'recording_start_to_odor_s': 3,
               'total_recording_s': 15}

projections, rois, df = ta.process_experiment(experiment_directory, \
                            substring2condition, stim_params)

"""
tplt.summarize_flies(df, projections)
tplt.summarize

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
    containing_blocks = fly_df[pd.notnull(fly_df[glom])].reset_index()['block']

    # TODO grid? units of seconds on x-axis. patch in stimulus presentation.
    # TODO check for onsets before we would expect them
    df = glom_df.reset_index()

    # plot individual traces
    g = sns.FacetGrid(df, hue='trial', col='odor', col_wrap=5)
    '''
    g = sns.FacetGrid(df, hue='trial', col='odor', col_wrap=5, palette=\
            sns.color_palette("Blues_d"))
    '''
    g = g.map(plt.plot, 'frame', glom, marker='.')

    g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
    title = glom + ' from blocks ' + str(containing_blocks.unique())
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.9)
    print(df.columns)

    # get set of blocks occurring with current odor (column name, the arg to lambda)
    f = lambda x: x + ' ' + str(list(filter(lambda e: type(e) is int, \
            set(containing_blocks.where(df['odor'] == x).unique()))))
    g.set_titles(col_func=f)


    # plot means w/ SEM errorbars
    df = glom_df.reset_index()
    df[glom] = df[glom].apply(pd.to_numeric)
    #grouped = glom_df.groupby(level=['odor', 'trial'])
    grouped = df.groupby(['odor', 'frame'])
    means = grouped.mean()
    means['sem'] = grouped[glom].sem()
    mdf = means.reset_index()
    g = sns.FacetGrid(mdf, col='odor', col_wrap=5)
    g = g.map(plt.errorbar, 'frame', glom, 'sem')

    g.set_ylabels(r'$\frac{\Delta{}F}{F}$')
    title = 'Mean w/ SEM for ' + glom + ' from blocks ' + str(containing_blocks.unique())
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.9)

if args.show_plots:
    plt.show()
"""

# TODO when i actually need p2o / stim order meta data, load that from within each dir

# TODO save computed df and images necessary for figures 
# (or figs separately maybe, and maybe not images?)
# and only recompute if code checksum differs from code checksum of last version

#pin2odors = [pickle.load(open(f)) for f in fix_names(p2o_prefix, pin2odor_names, '')]

#suffix = '.tif'

#glomeruli = ('dl5', 'dm6', 'vm7')

"""

# mock / butanone reared
for condition in sorted(files.keys()):
    for full, nick, all_p2o  in zip(files[condition], flies[condition], pin2odors[condition]):
        # for each odor panel (<=3 per fly, not all with same # of vials)

        p2o_dicts = []
        syncdata_files = []
        imaging_files = []

        '''
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
        '''
        for g in glomeruli:

            thorsync_file = glob.glob('/media/threeA/hong/flies/' \
                                + nick + o + '/SyncData*')[0] + '/Episode001.h5'
            imaging_file = full + o + suffix

            if args.print_summary_only:
                # TODO what was trial_duration used for?
                print_odor_order(thorsync_file, p2o_dict, imaging_file, trial_duration)

            p2o_dicts.append(p2o_dict)
            syncdata_files.append(thorsync_file)
            imaging_files.append(imaging_file)

        if not args.print_summary_only:
            fly_df = process_2p(imaging_files, syncdata_files, secs_before=3, secs_after=12, \
                    pin2odor=p2o_dicts)

            # TODO merge fly_dfs across flies. could probably also just use append.
            '''
            g = sns.FacetGrid(fly_df.reset_index(), hue='trial', col='odor', col_wrap=5)
            # TODO
            g = g.map(plt.plot, 'frame', glom)
            '''

"""
