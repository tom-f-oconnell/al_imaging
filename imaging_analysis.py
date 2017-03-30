#!/usr/bin/env python3

# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import tom.analysis as ta
import tom.plotting as tplt

plt.close('all')

###############################################################################################
# prepare to receive command line arguments
###############################################################################################

parser = argparse.ArgumentParser(description='Analysis pipeline for 2p movies of ' +\
        'GCaMP signals in the antennal lobe after odor presentation.',
        epilog='See github.com/tom-f-oconnell/al_imaging for more information.')

show_parser = parser.add_mutually_exclusive_group(required=False)
show_parser.add_argument('-s', '--showplots', dest='show_plots', action='store_true', \
        help='show plots at end')
show_parser.add_argument('-n','--noshowplots', dest='show_plots', action='store_false', \
        help='(default) do not show plots at end')

print_summary_parser = parser.add_mutually_exclusive_group(required=False)
print_summary_parser.add_argument('-p', '--printsummary', dest='print_summary_only', \
        action='store_true', help='output summary of experiments, including which odor was ' + \
        'presented during each frame range in each input TIF.')
print_summary_parser.add_argument('-a','--analysis', dest='print_summary_only', \
        action='store_false', help='(default) run analysis')

parser.add_argument('-t', '--test', dest='test', action='store_true')
# decide how these should interact w/ test
parser.add_argument('-c', '--check-data', dest='recheck_data', action='store_true')
parser.add_argument('-r', '--recompute', dest='recompute', action='store_true')
# TODO just have -s and -u be opposite, and imply u if not s?
parser.add_argument('-u', '--no-save-figs', dest='save_figs', action='store_false')

parser.set_defaults(show_plots=False, print_summary_only=False, test=False, \
        recheck_data=False, recompute=False, save_figs=True)

# automatically parses from sys.argv when called without arguments
args = parser.parse_args()


###############################################################################################
# call my analysis functions
###############################################################################################

# TODO include arg for directory to override envvar

if args.recompute == True:
    print('recomputing everything but registration model')

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
               'total_recording_s': 15}
               #'downsample_below_fps': 4}

# TODO only recompute if code checksum differs from code checksum of last version?
projections, rois, df = ta.process_experiment(experiment_directory, \
                            substring2condition, stim_params, cargs=args)

with open('experiment.output.p', 'wb') as f:
    pickle.dump((experiment_directory, stim_params, projections, rois, df), f)


###############################################################################################
# make plots
###############################################################################################

if not args.print_summary_only:
    # projections, and for each of {automated, manual} ROIs, plot ROIs on a grid
    # and stimuli x glomeruli means and traces
    # TODO environment var?
    save_to = None
    if args.save_figs:
        save_to = '/home/tom/figs/fly_summaries'

    tplt.summarize_flies(projections, rois, df, save_to=save_to)

    # a stimuli x glomeruli grid of traces for all automatically found ROIs
    # TODO and any manually identified traces with the glomerulus as a substring
    # (case insensitive)
    # TODO and include name of manual ROI on trace. maybe interactive like Remy's?
    if args.save_figs:
        save_to = '/home/tom/figs'

    tplt.summarize_experiment(df, save_to=save_to)

    if args.show_plots:
        plt.show()
    else:
        plt.close('all')

