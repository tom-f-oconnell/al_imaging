#!/usr/bin/env python3

# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import tom.analysis as ta

plt.close('all')

###############################################################################################
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
###############################################################################################

#sns.set_style('darkgrid')
#sns.set_palette('GnBu_d')

# TODO include arg for directory to overrid envvar

expdir_envvar = 'IMAGING_EXP_DIR'
if expdir_envvar in os.environ:
    experiment_directory = os.environ[expdir_envvar]

else:
    experiment_directory = '/home/tom/data/flies'
print(experiment_directory)

substring2condition = {'c': 'mock reared',
                       'e': '2-butanone 1e-4 reared'}

# variables in the Arduino code. other parameters will be loaded from
# ThorImage and ThorSync XML metadata, or pickle files describing stimuli
stim_params = {'ITI_s': 45,
               'odor_pulse_ms': 500,
               'repeats': 5,
               'recording_start_to_odor_s': 3,
               'total_recording_s': 15,
               'downsample_below_fps': 4}

projections, rois, df = ta.process_experiment(experiment_directory, \
                            substring2condition, stim_params, test=args.test)

with open('experiment.output.p', 'wb') as f:
    pickle.dump((experiment_directory, stim_params, projections, rois, df), f)


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
