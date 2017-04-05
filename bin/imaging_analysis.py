#!/usr/bin/env python3

# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from al_imaging import util
from al_imaging import analysis
from al_imaging import plotting

plt.close('all')

###############################################################################################
# prepare to receive command line arguments
###############################################################################################

parser = argparse.ArgumentParser(description='Analysis pipeline for 2p movies of ' +\
        'GCaMP signals in the antennal lobe after odor presentation.',
        epilog='See github.com/tom-f-oconnell/al_imaging for more information.')

parser.add_argument('-s', '--showplots', dest='show_plots', action='store_true', \
        help='show plots at end (will otherwise save them)')
parser.add_argument('-p', '--printsummary', dest='print_summary_only', \
        action='store_true', help='output summary of experiments, including which odor was ' + \
        'presented during each frame range in each input TIF.')

# TODO pick one randomly? w/ seed?
# decide how this should interact w/ others
parser.add_argument('-t', '--test', dest='test', action='store_true', \
        help='only runs on one subdirectory, to test changes faster')

parser.add_argument('-c', '--check-data', dest='recheck_data', action='store_true', \
        help='force rechecking which directories seem consistent. mostly uses ThorSync ' + \
        'data. only directories which pass this test will be analyzed.')

parser.add_argument('-r', '--recompute', dest='recompute', action='store_true', \
        help='recompute any values that might have been stored in the cache to save time. ' + \
        'does NOT imply -c')

# TODO include which relative dirs are used for [no-]save-figs and save-corrected
# maybe options to change them?
# TODO implement
parser.add_argument('-u', '--no-save-figs', dest='save_figs', action='store_false', \
        help='will not save figures. normally figures are saved to <TODO>')
parser.add_argument('-a', '--auto-glomeruli', dest='detect_glomeruli', action='store_true', \
        help='use "private" odors presented within a session to try segmenting glomeruli')
# TODO make highest priority. validate together with alternatives.
parser.add_argument('-i', '--input', dest='exp_dir', help='directory with subdirectories ' + \
        'containing data from single recording sessions')
parser.add_argument('-m', '--save-corrected', dest='save_motion_correction', action='store_true', 
        help='save the motion corrected TIFFs in <TODO>')

# just have -s and -u be opposite, and imply u if not s?

parser.set_defaults(show_plots=False, print_summary_only=False, test=False, \
        recheck_data=False, recompute=False, save_figs=True, detect_glomeruli=False, \
        save_motion_correction=False)

# automatically parses from sys.argv when called without arguments
args = parser.parse_args()


###############################################################################################
# call my analysis functions
###############################################################################################


# TODO actually implement in Memorize
if args.recompute == True:
    print('recomputing everything')

experiment_directory = util.get_expdir()
if not experiment_directory:
    experiment_directory = '/home/tom/data/flies'
print(experiment_directory)

# TODO pass this to plotting stuff? or use it to rename df condition
substring2condition = {'c': 'mock reared',
                       'e': '2-butanone 1e-4 reared'}

# TODO print these
# variables in the Arduino code. other parameters will be loaded from
# ThorImage and ThorSync XML metadata, or pickle files describing stimuli
stim_params = {'ITI_s': 45,
               'odor_pulse_ms': 500,
               'repeats': 5,
               'recording_start_to_odor_s': 3,
               'total_recording_s': 15}
               #'downsample_below_fps': 4}

projections, rois, df = analysis.process_experiment(experiment_directory, \
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

    plotting.summarize_flies(projections, rois, df, save_to=save_to)

    # a stimuli x glomeruli grid of traces for all automatically found ROIs
    # TODO and any manually identified traces with the glomerulus as a substring
    # (case insensitive)
    # TODO and include name of manual ROI on trace. maybe interactive like Remy's?
    if args.save_figs:
        save_to = '/home/tom/figs'

    plotting.summarize_experiment(df, save_to=save_to)

    if args.show_plots:
        plt.show()
    else:
        plt.close('all')

