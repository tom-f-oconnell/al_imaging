
# for backwards compability with Python 2.7
from __future__ import print_function
from __future__ import division

from . import plotting as tplt
from . import odors

import h5py
import scipy.io
import numpy as np
from os.path import join, split, getmtime, isdir, isfile, exists
import os
import xml.etree.ElementTree as etree
import re
import hashlib
import pickle

import tifffile

import thunder as td
from registration import CrossCorr
from registration.model import RegistrationModel
import cv2

import sys
import traceback
from joblib import Parallel, delayed

import pandas as pd
import plotly.offline as py
# TODO way to avoid needing this?
import plotly.graph_objs as go

import ijroi
from PIL import Image, ImageDraw


# taken from an answer by joeld on SO
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def is_anatomical_stack(d):
    return 'anat' in d


def get_fly_id(directory_name):
    # may fail if more underscores in directory name

    return '_'.join(split(directory_name)[-1].split('_')[:2])


def get_exptime(thorimage_dir):
    expxml = join(thorimage_dir, 'Experiment.xml')
    return int(etree.parse(expxml).getroot().find('Date').attrib['uTime'])


def get_synctime(thorsync_dir):
    syncxml = join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return getmtime(syncxml)


def get_readable_exptime(thorimage_dir):
    expxml = join(thorimage_dir, 'Experiment.xml')
    return etree.parse(expxml).getroot().find('Date').attrib['date']


# warn if has SyncData in name but fails this?
def is_thorsync_dir(d):
    if not isdir(d):
        return False
    
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


def is_imaging_dir(d):
    if not isdir(d):
        return False
    
    files = {f for f in os.listdir(d)}

    have_xml = False
    have_processed_tiff = False
    have_extracted_metadata = False
    for f in files:
        if 'Experiment.xml' in f:
            have_xml = True
        elif f == split(d)[-1] + '_ChanA.tif':
            have_processed_tiff = True
        elif f == 'ChanA_extracted_metadata.txt':
            have_extracted_metadata = True

    if have_xml and have_processed_tiff:
        if not have_extracted_metadata:
            print('WARNING: there does not seem to be extracted metadata in', d + '!!!')
        return True
    else:
        return False


def contains_thorsync(p):
    return True in {is_thorsync_dir(join(p,c)) for c in os.listdir(p)}


def contains_stimulus_info(d):
    has_pin2odor = False
    has_stimlist = False
    for f in os.listdir(d):
        if isfile(join(d,f)) and f == 'generated_pin2odor.p':
            has_pin2odor = True

        if isfile(join(d,f)) and f == 'generated_stimorder.p':
           has_stimlist = True

    return has_pin2odor and has_stimlist


def good_sessiondir(d):
    a = is_imaging_dir(d)

    if not a or is_anatomical_stack(d):
        return False

    b = contains_thorsync(d)
    c = contains_stimulus_info(d)
    return b and c


def parse_fly_id(filename):
    """
    Returns substring that describes fly / condition / block, and should be used to group
    all data externally. Used for generating indices for DataFrame and for matching data
    that came from the same experiment.
    """
    
    # TODO may have to change regex somewhat for interoperability with Windows
    match = re.search(r'(\d{6}_\d{2}(c|e)_o\d)', filename)
    return match.group(0)


def get_fly_info(filename):
    """
    Converts my date / flynumber / condition / block substring to three relevant variables.
    Combines date and flynumber into one string (fly_id).
    """
    #full_id = parse_fly_id(filename)
    full_id = '_'.join(split(filename)[-1].split('_')[:3])

    # contains date fly was run on, and order fly was run in that day
    # TODO this really works?
    fly_id = full_id[:-4]
    # last character of middle portion has this information
    condition = full_id.split('_')[1][-1]
    # assumes 1-9 sessions
    session = full_id[-1]

    return condition, fly_id, session


def load_frame(frame_filename):
    #return bioformats.load_image(frame_filename)
    return td.images.fromtif(frame_filename).toarray()


def one_d(nd):
    """
    Returns a one dimensional numpy array, for input with only one non-unit length dimension.
    """
    return np.squeeze(np.array(nd))


# TODO more idiomatic way to do this?
# remove?
def df_add_col(df, col_label, col):
    """
    Returns a pandas.DataFrame with the data in col inserted in a new column.

    If df is None, returns a new DataFrame with one column.
    Does not handle the case of an empty DataFrame.
    """

    tmp_df = pd.DataFrame({col_label: col})
    df = pd.concat([df, tmp_df], axis=1)

    return df


def sumnan(array):
    """ utility for faster debugging of why nan is somewhere it shouldn't be """
    return np.sum(np.isnan(array))


def get_thorimage_metafile(session_dir):
    return join(session_dir, 'Experiment.xml')


def get_thorsync_metafile(session_dir):
    subdirs = [join(session_dir,d) for d in os.listdir(session_dir)]
    maybe = [d for d in subdirs if is_thorsync_dir(d)]

    if len(maybe) != 1:
        raise AssertionError('too many possible ThorSync data directories in ' + session_dir + \
                ' to automatically get XML metadata. fix manually.')
    else:
        return join(maybe[0], 'ThorRealTimeDataSettings.xml')


def get_thorsync_hdf5(session_dir):
    subdirs = [join(session_dir,d) for d in os.listdir(session_dir)]
    maybe = [d for d in subdirs if is_thorsync_dir(d)]

    if len(maybe) != 1:
        raise AssertionError('too many possible ThorSync data directories in ' + session_dir + \
                ' to automatically get XML metadata. fix manually.')
    else:
        d = maybe[0]
        num_h5 = 0
        for f in os.listdir(d):
            if '.h5' in f:
                num_h5 += 1

        if num_h5 > 1:
            print('WARNING: more than one EpisodeXXX.h5. could have picked wrong one!')

        return join(maybe[0], 'Episode001.h5')


def check_consistency(d, stim_params):
    print(bcolors.BOLD + bcolors.OKBLUE + 'Checking:\n' + d + bcolors.ENDC)

    with open(join(d,'generated_pin2odor.p'), 'rb') as f:
        connections = pickle.load(f)
        pin2odor = dict(map(lambda x: (x[0], x[1]), connections))

    with open(join(d,'generated_stimorder.p'), 'rb') as f:
        pins = pickle.load(f)

    imaging_metafile = get_thorimage_metafile(d)
    thorsync_metafile = get_thorsync_metafile(d)

    actual_fps = get_effective_framerate(imaging_metafile)
    averaging = get_thor_averaging(imaging_metafile)
    samprate_Hz = get_thorsync_samprate(thorsync_metafile)

    # variables from Arduino code determining trial structure
    scopeLen = stim_params['total_recording_s']
    onset = stim_params['recording_start_to_odor_s']
    odor_pulse_len = stim_params['odor_pulse_ms'] / 1000.0

    # this is how i defined it before: onset to onset, not interval between recordings
    # may way to change
    # (scopeLen + time not recording after each trial = 15 + 30 seconds)
    iti = stim_params['ITI_s']

    arr = td.images.fromtif(join(d, split(d)[-1] + '_ChanA.tif')).toarray().squeeze()
    num_frames = arr.shape[0]

    def err():
        print('\n' + bcolors.FAIL, end='')
        traceback.print_exc(file=sys.stdout)
        print('skipping!' + bcolors.ENDC + '\n')
        return False

    # TODO i don't actually use the # repeats now i think, but i could
    # TODO do things that don't require decoding before those that do

    num_trials = len(pins)

    try:
        df = load_thor_hdf5(d, exp_type='imaging')

        real_pins, odor_pulses = decode_odor_used(df['odor_used'], samprate_Hz)

        if pins != real_pins:
            print(pins)
            print(real_pins)
            print(len(pins))
            print(len(real_pins))

        assert pins == real_pins, 'decoding did not match saved pins'
        df['odor_pulses'] = odor_pulses

    # file probably didn't exist
    except OSError:
        return err()
    except AssertionError:
        return err()

    try:
        # TODO call this from decode_odor_used instead?
        # count pins and make sure there is equal occurence of everything to make sure
        # trial finished correctly
        check_equal_ocurrence(pins)
        check_acquisition_triggers_crude(df['acquisition_trigger'], len(pins), samprate_Hz)

        # TODO might want to factor this back into below. dont need?
        est_fps = num_frames / (scopeLen * len(pins))

        # asserts framerate is sufficiently close to expected
        check_framerate_est(actual_fps, est_fps)

        #check_duration_est(df['frame_counter'], actual_fps, averaging, \
        #        num_frames, pins, scopeLen, verbose=True)
        check_frames_per_trigger(df['acquisition_trigger'], df['frame_counter'], \
                averaging, num_frames, samprate_Hz)

        check_acquisition_time(df['acquisition_trigger'], scopeLen, samprate_Hz)

        odor_pulse_len = stim_params['odor_pulse_ms'] / 1000

        check_odorpulse(df['odor_pulses'], odor_pulse_len, samprate_Hz, epsilon=0.05)

        check_odor_onset(df['acquisition_trigger'], df['odor_pulses'], \
                onset, samprate_Hz, epsilon=0.05)

        check_iti(df['odor_pulses'], iti, samprate_Hz, epsilon=0.05)

        check_iti_framecounter(df['frame_counter'], iti, scopeLen, samprate_Hz, epsilon=0.3)

        # don't need this now that i have a method to toss extra frames
        #assert num_frames % num_trials == 0, 'frames do not evenly divide into trials.\n\n' + \
        #        str(num_frames) + ' frames and ' + str(num_trials) + ' trials.'

    except AssertionError:
        down = 1000
        data = [go.Scatter(x=df.index[::down] / samprate_Hz, y=df.acquisition_trigger[::down],\
                    name='Acquisition Trigger')
               ,go.Scatter(x=df.index[::down] / samprate_Hz, y=np.diff(df.frame_counter[::down]),\
                    name='Changes in frame counter')
               ,go.Scatter(x=df.index[::down] / samprate_Hz, y=df.odor_used[::down],
                    name='Odor pulse + odor signalling')
               ,go.Scatter(x=df.index[::down] / samprate_Hz, y=df.odor_pulses[::down],
                    name='Odor pulse')
               ]

        layout = dict(title='Acquisition trigger during session '+split(d)[-1], \
                yaxis=dict(title='Voltage'),xaxis=dict(title='Time', rangeslider=dict()))

        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='sync_debugging_' + split(d)[-1] + '.html')

        return err()

    print(bcolors.BOLD + bcolors.OKGREEN+ 'all checks passed!' + bcolors.ENDC + '\n')
    return True


def load_thor_hdf5(fname, exp_type='pid'):
    if os.path.isdir(fname):
        fname = get_thorsync_hdf5(fname)

    f = h5py.File(fname, 'r')

    if exp_type == 'pid':
        return pd.DataFrame({'odor_used':  one_d(f['AI']['odor_used']), \
                             'ionization': one_d(f['AI']['ionization_detector'])})

    elif exp_type == 'imaging':
        # WARNING: this will load each of these time series in full into memory
        # performance may suffer because of it, and it would be possible to write code
        # that doesn't do that
        # (the np.array(.) step is the bad one)

        # there are some other variables in the hdf5 file that we don't care about
        return pd.DataFrame({'odor_used':           one_d(f['AI']['odor_used']), \
                             'acquisition_trigger': one_d(f['AI']['acquisition_trigger']), \
                             'frame_counter':       one_d(f['CI']['frame_counter'])})

    else:
        assert False, 'exp_type not valid'


# TODO profile. i suspect this is adding a fair bit of time.
def load_data(name, exp_type=None):

    assert not exp_type is None, 'must specify an exp_type'

    save_prefix = '/media/threeA/hong/.tmp/'

    if name[-3:] == '.h5':
        # TODO dont make this assumption! load from thorsync xml file
        samprate_Hz = 30000

        df = load_thor_hdf5(name, exp_type)

        nick = hashlib.md5(name.encode('utf-8')).hexdigest()[:10]

        # this step seems like it might be taking a lot longer now that i switched to pandas?
        if (not exists(save_prefix + nick + '_odors_pulses.npy') \
                or not exists(save_prefix + nick + '_pins.p')):

            if not exists(save_prefix):
                os.mkdir(save_prefix)

            pins, odor_pulses = decode_odor_used(df['odor_used'], samprate_Hz)

            # axis=1 is necessary to drop a whole column
            # (i feel it should be obvious if rows are only indexed by integers though...)
            df.drop('odor_used', axis=1, inplace=True)

            odor_pulses_df = pd.DataFrame({'odor_pulses': np.squeeze(odor_pulses)})
            df = pd.concat([odor_pulses_df, df], axis=1)

            np.save(save_prefix + nick + '_odors_pulses', odor_pulses)
            with open(save_prefix + nick + '_pins.p', 'wb') as f:
                pickle.dump(pins, f)

        else:
            df.drop('odor_used', axis=1, inplace=True)

            odor_pulses = np.load(save_prefix + nick + '_odors_pulses.npy')

            odor_pulses_df = pd.DataFrame({'odor_pulses': np.squeeze(odor_pulses)})
            df = pd.concat([odor_pulses_df, df], axis=1)

            with open(save_prefix + nick + '_pins.p', 'rb') as f:
                pins = pickle.load(f)

    elif name[-4:] == '.mat' and exp_type == 'pid':
        samprate_Hz = 200

        data = scipy.io.loadmat(name)['data']

        # TODO convert to dataframe

        if data.shape[1] < 2:
            print(name + ' did not appear to have ionization data. Skipping.')
            return
            # TODO ? or is it missing something else? valve? PLOT

        # at 200Hz
        odor_pulses = data[:int(data[:,1].shape[0]),1]
        ionization = data[:int(data[:,0].shape[0]),0]
    
        # there is not something that can vary as on the olfactometer
        pins = None
    else:
        assert False, 'not sure how to load data'

    '''
    if exp_type == 'pid':
        return odor_pulses, pins, ionization, samprate_Hz

    elif exp_type == 'imaging':
        return acquisition_trig, odor_pulses, pins, frame_counter, samprate_Hz
    '''
    # i'm going to have to index pins and df separately for now

    return df, pins, samprate_Hz


# TODO fix places that call this. docstring
def load_pid_data(name):
    return load_data(name, exp_type='pid')


# TODO docstring
def load_2p_syncdata(name):
    return load_data(name, exp_type='imaging')


def frame_filenames(directory):
    # sorting should be consistent w/ numbering? test
    # ChanB should have the fiduciary (which wasn't actually working for any flies so far)
    return sorted([join(directory,d) for d in os.listdir(directory) \
            if not 'Preview' in d and '.tif' in d and 'ChanA' in d])


def decode_odor_used(odor_used_analog, samprate_Hz, verbose=True):
    # TODO update docstring w/ mixture protocol
    """ 
    At the beginning of every trial, the Arduino will output a number of 1ms wide high pulses
    equal to the pin number it will pulse on that trial.

    @input: analog trace (sampled at samprate_Hz) output by Arduino odor_signalling pin
    (also labelled as odor broadcast)

    @output: a list with a number of elements equal to the number of trials, and with each
    element holding the pin number pulsed during the trial with that index

    WARNING: This signalling method should work find on pins 1 through 13 (though you 
    probably don't want to use pin 1, because along with pin 0 it is used (on most 
    Arduinos?) for Serial communication, BUT this will likely not work on the analog pins
    and it will definitely not work on pin zero.

    If you have an Arduino Mega or comparable, you will need to update the max_pin_number
    to whichever you actually use.
    """
    def decode(odor_used_array):
        tolerance = 0.10
        pulse_width = 0.001 # sec (1 ms)
        # between pins signalled within one group. pins are toggled within a few microseconds
        # when odor is presented.
        between_pins_s = 0.011

        pulse_samples = pulse_width * samprate_Hz
        between_samples = between_pins_s * samprate_Hz

        # need to change if using something like an Arduino Mega with more pins
        # not sure how analog pins would be handled as is? (not that I use them)
        # TODO fix pin 0 or just assert that it can not be used
        # (how to force something to have a compile time error in Arduino?)
        min_pin_number = 4
        max_pin_number = 13

        # factor of 2 because the Arduino sends the pin low for pulse_width between each high period
        # of duration pulse_width
        # TODO do i actually need this?
        max_signaling_samples = int(round(2 * max_pin_number * pulse_samples * (1 + tolerance)))

        # to exclude false positives from very short transients or data acquisition errors
        pulse_samples_min = int(round(pulse_samples * (1 - tolerance)))

        # the same pin will signal when go high (almost exactly--within microseconds of) when the
        # actual valve pin does, so we also need to limit our search to short pulses
        pulse_samples_max = int(round(pulse_samples * (1 + tolerance)))

        # may need to adjust these, or set a separate tolerance
        # i think i am consistently underestimating them
        between_samples_min = int(round(between_samples * (1 - tolerance)))
        between_samples_max = int(round(between_samples * (1 + tolerance)))

        # TODO need this?
        total_samples = odor_used_analog.shape[0]

        onsets, offsets = threshold_crossings(odor_used_array)

        odor_pins = []
        signal_onsets = []
        signal_offsets = []
        current_set = set()
        counter = 0
        last_off = None

        def valid_pulse(ta, tb):
            assert ta < tb, 'order doesnt make sense'
            delta = tb - ta
            return delta >= pulse_samples_min and delta <= pulse_samples_max

        def between_pins(ta, tb):
            assert ta < tb, 'order doesnt make sense'
            delta = tb - ta
            return delta >= between_samples_min and delta <= between_samples_max

        for on, off in zip(onsets, offsets):
            if valid_pulse(on, off):
                if last_off is not None and between_pins(last_off, on):
                    current_set.add(counter)
                    counter = 0

                if last_off is None or (on - last_off) > between_samples_max:
                    signal_onsets.append(on - 1)

                counter += 1

            #?
            # on=start of (for me, 500ms) odor presentation, and off=end of it
            # because that'll be the first pulse after the signalling
            # and the next pulse should be a valid_pulse again
            elif last_off is not None:
                current_set.add(counter)

                signal_offsets.append(last_off + 1)
                odor_pins.append(current_set)

                current_set = set()
                counter = 0

            last_off = off

        return tuple(odor_pins), signal_onsets, signal_offsets

    odor_used_array = np.array(odor_used_analog)
    pins, onsets, offsets = decode(odor_used_array)

    for on, off in zip(onsets, offsets):
        odor_used_array[max(on-1,0):min(off+1,odor_used_array.shape[0] - 1)] = 0

    counts = dict()
    for p in pins:
        if type(p) is set:
            p = frozenset(p)
        counts[p] = 0
    for p in pins:
        if type(p) is set:
            p = frozenset(p)
        counts[p] = counts[p] + 1

    # TODO just call the check that does this?
    if len(set(counts.values())) != 1:
        print('Warning: some pins seem to be triggered with different frequency')

    return pins, odor_used_array


def ecdf(data):
    return np.sort(data), np.arange(1, len(data)+1) / len(data)


def bs_resample(data, iters=100000, f=np.mean):
   """ Bootstrap resamples the statistic determined by function f (np.mean by default)
   iters times and returns an array of that length. """
   bs_reps = np.empty(iters)

    # Compute replicates
   for i in range(iters):
       bs_sample = np.random.choice(data, size=len(data))
       """ Could do a replicate of any statistic we wanted:
       CV, std, artibtrary percentiles, whatever, but here we are doing a 
       bootstrap estimate of the mean.
       """
       bs_reps[i] = f(bs_sample)

   return bs_reps


def confidence_intervals(profiles, tail=5):

    ci_lowers = []
    ci_uppers = []
    for i in range(profiles.shape[1]):
        bs_reps = bs_resample(profiles[:,i], iters=1000) 
        cis = np.percentile(bs_reps, [tail, 100 - tail])
        ci_lowers.append(cis[0])
        ci_uppers.append(cis[1])

    return np.array(ci_lowers), np.array(ci_uppers)


def simple_onset_windows(num_frames, num_trials):
    """
    Returns tuples of (start_index, end_index), jointly spanning all num_frames, if num_frames
    can be divided cleanly into num_trials.

    Otherwise will fail an assertion.
    """

    print(bcolors.WARNING + 'WARNING: WINDOWS NOT CURRENTLY CALCULATED USING ' + \
            'secs_before OR secs_after' + bcolors.ENDC)

    frames_per_trial = int(num_frames / num_trials)

    #assert np.isclose(num_frames / num_trials, round(num_frames / num_trials)), \
    #        'num_frames not divisible by num_trials'

    windows = []

    for n in range(num_trials):
        offset = n * frames_per_trial
        # TODO test whether this will result in the correct NON OVERLAPPING windows!!
        windows.append((offset, offset + frames_per_trial))

    return tuple(windows)


# TODO TODO this can extend beyond length of imaging_data!!! why??? seem to drift relative to simple

# TODO could probably speed up decoding function by vectorizing similar to below?
# most time above is probably spent just finding these crossings
# TODO could just start windows @ frame trigger onset?
def onset_windows(trigger, secs_before, secs_after, samprate_Hz=30000, frame_counter=None, \
        acquisition_trig=None, threshold=2.5, max_before=None, max_after=None, averaging=15,\
        actual_fps=None):
    """ Returns tuples of (start, end) indices in trigger that flank each onset
    (a positive voltage threshold crossing) by secs_before and secs_after.

    -length of windows will all be the same (haven't tested border cases, but otherwise OK)
    -trigger should be a numpy array that is crosses from < to > threshold locked with what we want
    -secs_before and secs_after both relative to onset of the pulse.

    WARNING: this function is not yet equipped to deal with data that actually has full flyback
    frames
    """
    # TODO just diff > thresh...? compare
    shifted = trigger[1:]
    truncated = trigger[:-1]
    # argwhere and where seem pretty much equivalent with boolean arrays?
    onsets = np.where(np.logical_and(shifted > threshold, truncated < threshold))[0]

    # checking how consistent the delays between odor onset and nearest frames are
    # WARNING: delays range over about 8 or 9 ms  --> potential problems (though generally faster
    # than we care about) (ACTUALLY POTENTIALLY GREATER BECAUSE THOSE ARE RAW FRAMES)
    # TODO how to convert these to effective frames
    '''
    frames = list(map(lambda x: int(frame_counter[x]), onsets))
    # get the index of the last occurence of the previous frame number
    # to compare to index of odor onset
    previous = list(map(lambda f: np.argwhere(np.isclose(frame_counter, f - 1))[-1,0], frames))
    assert len(previous) == len(onsets)
    delays = list(map(lambda x: x[1] - x[0], zip(onsets, previous)))
    print(delays)
    '''

    # TODO so is most variability in whether or not last frame is there? or first?
    # matters with how i line things up in averaging... (could easily be 500 ms difference)

    if not max_before is None and not max_after is None:
        assert secs_before <= max_before, \
                'trying to display more seconds before onset than available'

        assert secs_after <= max_after , 'only recorded ' + str(max_before+max_after) + 's per trial'

    # asserts that onsets happen in a certain position in trial with a certain
    # periodicity. might introduce too much risk of errors.

    # TODO see if negative dF/F are evident with this set of onsets
    '''
    curr = 0
    i = 0
    onsets = []

    # TODO account for lag or will probably underestimate true evoked fluorescence
    while curr < len(trigger):
        curr = (max_before +  (max_before + max_after) * i) * samprate_Hz
        i += 1
        # TODO check
        onsets.append(int(round(curr)))
    '''

    shifted = acquisition_trig[1:]
    truncated = acquisition_trig[:-1]

    # should be same threshold for this signal as for odor pulses
    acq_onsets = np.where(np.logical_and(shifted > threshold, truncated < threshold))[0]
    acq_offsets = np.where(np.logical_and(shifted < threshold, truncated > threshold))[0]

    # this won't be true if the signal starts and ends in different states, but it shouldn't
    assert acq_onsets.shape == acq_offsets.shape, 'different # of acquisition trigger ' + \
            'onsets and offsets. comment this assert if some electrial issue caused ' + \
            'experiment to start or finish with an anomalous voltage on that line.'

    acquisition_times = []
    raw_frames_in_acqtimes = []
    frame_counter_max = []

    for start, stop in zip(acq_onsets, acq_offsets):
        acquisition_times.append(int(stop - start))
        raw_frames_in_acqtimes.append(np.sum(np.diff(frame_counter[start:stop], axis=0) > 0))
        frame_counter_max.append(int(frame_counter[stop]))

    # is this all the frames we have?
    # is frame_counter ever at its max while acq_trig is HIGH?
    # the latter should be necessary for the former
    # TODO is the difference between this and the max_frame_counter a multiple of len(pins)?
    print(sum(raw_frames_in_acqtimes), 'raw frames signaled while supposed to be acquiring')
    print(frame_counter_max)
    print(np.diff(frame_counter_max), 'diff frame_counter')
    print(list(map(lambda x: x / averaging, np.diff(frame_counter_max))), ' / averaging')
    print(raw_frames_in_acqtimes, 'raw frames in acqtimes')
    print(list(map(lambda x: x / averaging, raw_frames_in_acqtimes)), ' / averaging')
    print('')

    #print(acquisition_times)

    acq_max = max(acquisition_times)
    acq_min = min(acquisition_times)
    acqtime_range = acq_max - acq_min

    assert acqtime_range <= 50, \
            'acquisition trigger is HIGH for durations that vary over more than 1 frame at ' +\
            'the ThorSync sampling frequency. range=' + str(acqtime_range) + ' indices at '+\
            str(samprate_Hz) + 'Hz'

    # plots frame trigger, frame counter, and odor onset from 1s before acq trigger 
    # onset to 1s after odor onset
    # TODO same for end for all of them?
    '''
    start = acq_onsets[0] - samprate_Hz * 1
    end = onsets[0] + samprate_Hz * 1
    indices = np.arange(start, end)
    plt.plot(indices, acquisition_trig[start:end], label='acquisition trigger')
    plt.plot(indices, frame_counter[start:end], label='frame counter')
    plt.plot(indices, trigger[start:end], label='valve pulse')
    plt.legend()
    plt.show()
    '''

    if frame_counter is None:
        return list(map(lambda x: (x - int(round(samprate_Hz * secs_before)), x + \
                int(round(samprate_Hz * secs_after))), onsets))
    else:
        windows = list(map(lambda x: ( int(round(int(frame_counter[x - \
                int(round(samprate_Hz * secs_before))]) / averaging)), \
                int(round(int(frame_counter[x + \
                int(round(samprate_Hz * secs_after))]) /  averaging)) ), onsets))

        # if we use all frames recorded before onset, we should use the first frame (0 index)
        if secs_before == max_before:
            assert windows[0][0] == 0, 'using all frames surrounding trial start, yet not ' + \
                'starting on first frame. instead, frame: ' + str(windows[0][0])

        return windows


# TODO test
def average_within_odor(deltaF, odors):
    """ 
    Returns a dict of (odor name: triggered average of deltaF for that odor)
    -each value of the dict will still be of dimensions (MxNxT), just averaged across
     however many presentations of the odor there were

    -deltaF is a tuple of numpy arrays ( ((MxN image)xT), ...) that are the regions after onset
     , already subtracted from the baseline frames in the (currently 1s) before
    -we just need to average the set of deltaF and pins elements whose indices correspond
    """

    odor2avg = dict()

    unique_odors = set(odors)
    for o in unique_odors:

        # the filter gets pairs (of index *aligned* lists) where second element is what we want
        # the map just throws out the second element, because we actually want the matching first
        odor2avg[o] = np.mean(np.stack(map(lambda x: x[0], \
            filter(lambda x: x[1] == o, zip(deltaF, odors)))), axis=0)

    return odor2avg


def project_each_value(data_dict, f):
    """ 
    For applying different functions for compressing data along one axis of the averages.
    The values of data_dict will generally be averaged delta F / F numpy arrays.
    """

    return {key: f(value) for key, value in data_dict.items()}


def odor_triggered_average(signal, windows, pins):
    """ Returns a dict of (pin: triggered average of signal for that pin)

    -windows is a list of tuples of (start_index, end_index)
    -the indices of pins and windows should match up (so the ith entry in pins is the pin that was 
     used on the trial selected by the index pair in the ith entry in windows)
    """

    if not pins is None:
        unique_pins = set(pins)
        pin2avg = dict()

        for p in unique_pins:

            # get windows for trials using current pin only
            curr_windows = list(map(lambda x: x[1], filter(lambda x: x[0] == p, zip(pins, windows))))

            # calculate the mean of the signal in all of the above
            pin2avg[p] = np.mean(np.squeeze(np.stack(map(lambda x: signal[x[0]:x[1]], \
                    curr_windows))), axis=0)

        return pin2avg

    else:
        # kinda hacky
        dummy = dict()
        dummy[-1] = np.mean(np.squeeze(np.stack(map(lambda x: signal[x[0]:x[1]], windows))), axis=0)
        return dummy


def generate_motion_correction(directory, images):
    """
    Aligns images within a movie of one plane to each other. Saves the transform.
    """
    print('starting thunder registration...')
    print(directory)

    # TODO seemed to not be saving...

    fname = '.thunder_registration_model.p'
    cache = join(directory, fname)

    print(cache)

    if isfile(cache):
        with open(cache, 'rb') as f:
            saved = pickle.load(f)
            return saved
    else:
        print('cache didnt exist')

    # registering to the middle of the stack, to try and maximize chance of a good registration
    middle = round(images.shape[0] / 2.0)

    # TODO implement w/ consistent interface
    if type(images) is not np.ndarray:
        reference = images[middle,:,:].toarray().squeeze()
    else:
        reference = images[middle,:,:].squeeze()

    reg = CrossCorr()
    model = reg.fit(images, reference=reference)

    with open(cache, 'wb') as f:
        pickle.dump(model, f)

    print('done')
    return model


def correct_and_write_tif(directory, transforms=None):
    """
    Will NOT work with TIFs too large to hold in memory at once.
    """

    # TODO load model if it is None
    print('applying transformation...')

    files = frame_filenames(directory)
    first = load_frame(files[0])

    #registered = np.zeros((*first.shape, len(files)))
    registered = np.empty(( *first.shape, len(files))) * np.nan
    # check no nan?

    for i, f in enumerate(files):
        print(str(i) + '/' + str(len(files)), end='        \r')
        frame = load_frame(f)
        # TODO add support for making RegistrationModels from matrices
        #registered[:,:,i] = RegistrationModel({(0,): transforms[i,:]}).transform(frame)
        model = transforms[f]
        registered[:,:,i] = model.transform(frame).toarray()

        '''
        tplt.hist_image(registered[:,:,i])
        plt.figure()
        plt.imshow(registered[:,:,i])
        plt.show()
        '''

    # TODO deal with OMEXML metadata too?
    
    # TODO test this is readable and is identical to data before writing
    reg_file_out = split(directory)[-1] + '_registered.bf.tif'
    path_out = join(directory, reg_file_out)
    print('saving registered TIF to', path_out, '...', end='')

    normalized = cv2.normalize(registered, None, 0.0, 2**16 - 1, cv2.NORM_MINMAX).astype(np.uint16)

    expanded = np.expand_dims(normalized, 0)
    # this empirically seems to be correct, but apparently goes against
    # online documentation
    expanded = np.swapaxes(expanded, 2, 3)
    expanded = np.swapaxes(expanded, 1, 2)
    tifffile.imsave(path_out, expanded, imagej=True)

    #bioformats.write_image(path_out, registered, bioformats.PT_UINT16, size_t=len(files))

    print('done.')


def xml_root(xml):
    tree = etree.parse(xml)
    return tree.getroot()


def get_thorsync_samprate(thorsync_metafile):
    root = xml_root(thorsync_metafile)
    rate = None
    # */*/ means only search in all great-great-grandchildren of root
    for node in root.findall('*/*/SampleRate'):
        if node.attrib['enable'] == '1':
            curr_rate = int(node.attrib['rate'])
            if rate is not None:
                if rate != curr_rate:
                    raise ValueError('sampling rates across acquisition boards differ. ' + \
                        'you will need to specify which you are using.')

            else:
                rate = curr_rate

    if rate is None:
        raise ValueError('did not find information on sampling rate in ' + thorsync_metafile)

    return rate


def get_effective_framerate(imaging_metafile):
    """
    Args:
        imaging_metafile: the location of the relevant Experiment.xml file

    Returns:
        the frame rate recorded in imaging_metafile by ThorImageLS.
    """
    root = xml_root(imaging_metafile)
    lsm_node = root.find('LSM')

    if lsm_node.attrib['averageMode'] == '1':
        actual_fps = float(lsm_node.attrib['frameRate']) / int(lsm_node.attrib['averageNum'])
    elif lsm_node.attrib['averageMode'] == '0':
        actual_fps = float(lsm_node.attrib['frameRate'])

    return actual_fps


def get_thor_averaging(imaging_metafile):
    """
    Args:
        imaging_metafile: the location of the relevant Experiment.xml file

    Returns:
        the number of frames averaged into a single effective frame in the output TIF, as 
        recorded in the input file by ThorImageLS
    """
    root = xml_root(imaging_metafile)
    lsm_node = root.find('LSM')
    return int(lsm_node.attrib['averageNum'])


def get_thor_notes(imaging_metafile):
    """
    Args:
        imaging_metafile: the location of the relevant Experiment.xml file

    Returns:
        the number of frames averaged into a single effective frame in the output TIF, as 
        recorded in the input file by ThorImageLS
    """
    root = xml_root(imaging_metafile)
    return root.find('ExperimentNotes').attrib['text']


def check_acquisition_triggers_crude(acquisition_trigger, target, \
        samprate_Hz, minimum_high_time_s=1):

    print('Number of observed triggers (debounced)... ', end='')
    onsets, offsets = threshold_crossings(acquisition_trigger)

    deliberate_triggers = 0
    for on, off in zip(onsets, offsets):
        assert off > on

        if (off - on) / samprate_Hz > minimum_high_time_s:
            deliberate_triggers += 1

    assert deliberate_triggers == target, bcolors.FAIL + 'did not trigger acquisition as many ' + \
            'times as you thought, or experiment was stopped prematurely\ntarget: ' + str(target) + \
            '\ndetected: ' + str(deliberate_triggers) + bcolors.ENDC

    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)
    return True


def check_equal_ocurrence(items):
    """
    Asserts each item that occurs in input dictionary does so with the same frequency as other. 
    May suggest experiment did not run properly.

    Args:
        items (dict)
    """

    d = dict()

    for i in items:
        if type(i) == set:
            i = frozenset(i)

        if not i in d:
            d[i] = 1
        else:
            d[i] += 1

    assert len(set(d.values())) == 1, 'some valves seem to have been triggered more than others'

    return True


def check_framerate_est(actual_fps, est_fps, epsilon=0.5, verbose=False):
    """
    Asserts framerate from ThorImage metadata is close to what we expect
    """

    print('Estimated framerate... ', end='')

    if verbose:
        print('estimated frame rate (assuming recording duration)', est_fps, 'Hz')
        print('framerate in metadata', actual_fps, 'Hz')

    assert abs(est_fps - actual_fps) < epsilon, 'expected and actual framerate mismatch'

    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_duration_est(frame_counter, actual_fps, averaging, num_frames, pins, scopeLen,
        verbose=False, epsilon=0.4):
    """
    Checks we have about as many frames as we'd expect.

    Prints more debugging information with `verbose` keyword argument set to True.
    """

    print('Recording duration... ', end='')

    mf = np.max(frame_counter)

    '''
    if verbose:
        print('actual_fps', actual_fps)
        print('averaging', averaging)
        print('actual frames in series', num_frames)
        print('number of odor presentations', len(pins))
        print('target recording duration per trial', scopeLen)
        print('')

        # some of these are redundant with each other
        print(mf / averaging / actual_fps, 'seconds implied (at actual framerate), ' + \
                "assuming all frames in counter contribute to an average (they don't)")
        print(len(pins) * scopeLen, 'expected number of seconds (# trials * length of trial)')
        print(num_frames / actual_fps, \
                'actual # frames / actual_framerate = actual total recording duration')
        print('')

        print(num_frames * averaging, \
                'expected max value of frame_counter, given number of frames in the TIF, and ' + \
                'assuming all frames that increment frame_count contribute to an average')
        print(mf, 'actual maximum frame_counter value')
        print('')

        print('the above, divided by number of odor presentations detected')
        print('raw frames per odor presentation')
        print(num_frames * averaging / len(pins), 'expected')
        print(mf / len(pins), 'actual')
        print('')

        print('effective frames per odor presentation')
        print(num_frames / len(pins), 'actual')
        print(mf / len(pins) / averaging, 'from frame_counter')
        print('')

        # TODO count raw frames per acquisition trigger high to see if they are all at the 
        # end or something. would hint at how to handle
    '''

    # TODO describe magnitude of potential error
    if abs(num_frames * averaging - mf) >= (actual_fps / 3):
        print(bcolors.WARNING + 'WARNING: not all raw frames correspond to effective frames.')
        print('Potential triggering / alignment problems.')
        print(bcolors.ENDC)

    msg = 'expected a different # of effective frames.'
    if verbose == False:
        msg += ' set verbose=True in ' + \
           'check_duration_est arguments for some quick troubleshooting information'

    # TODO TODO i do need to figure out where the mismatch between counted and effective frames
    # comes from (accounting for averaging)
    # TODO does the # of extra frames in max_frame diff in windows account for 
    # this discrepancy? up to maybe an extra multiple? !!

    assert abs(num_frames - (mf / averaging)) < epsilon, msg + \
            '\n' + str(num_frames)
           
    #print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_frames_per_trigger(acquisition_trigger, frame_counter, averaging, num_frames, samprate_Hz):
    
    print('Checking frames per trigger...')
    tonsets, toffsets = threshold_crossings(acquisition_trigger)

    in_trigger = []
    in_trigger_floor = []

    # check trigger windows extending 1s beyond each
    beyond_trigger = []
    beyond_trigger_floor = []

    for on, off in zip(tonsets, toffsets):
        in_trigger.append(np.sum(np.diff(frame_counter[on:off])) / averaging)
        in_trigger_floor.append(np.sum(np.diff(frame_counter[on:off])) // averaging)

        beyond_trigger.append(np.sum(np.diff(frame_counter[on:off + samprate_Hz])) / averaging)
        beyond_trigger_floor.append(np.sum(np.diff(frame_counter[on:off + samprate_Hz])) //averaging)

    '''
    print(sum(in_trigger_floor))
    print(sum(beyond_trigger_floor))
    print(num_frames)
    '''
    assert num_frames == sum(beyond_trigger_floor)
    # TODO print ok


# TODO use this function in other appropriate instances
def threshold_crossings(signal, threshold=2.5):
    """
    Returns indices where signal goes from < threshold to > threshold as onsets,
    and where signal goes from > threshold to < threshold as offsets.
    
    Cases where it at one index equals the threshold are ignored. Shouldn't happen and 
    may indicate electrical problems for our application.
    """

    # TODO could redefine in terms of np.diff
    # might be off by one?
    shifted = signal[1:]
    truncated = signal[:-1]

    # watch for corner case... (I don't want signals slowly crossing threshold though)
    onsets = np.where(np.logical_and(shifted > threshold, truncated < threshold))[0]
    offsets = np.where(np.logical_and(shifted < threshold, truncated > threshold))[0]

    return onsets, offsets


def check_onset2offset(signal, target, samprate_Hz, epsilon=0.005, msg=''):
    """
    Generic function to check pulse length of a signal is appropriate.

    Args:
        signal (np.ndarray)
        target (float): how many seconds pulse (HIGH) duration should be
        samprate_Hz: sampling rate at which signal was acquired
        epsilon (float): absolute deviation in seconds allowed from target before assertion fails
        msg (str): message to include in AssertionError output
    """

    onsets, offsets = threshold_crossings(signal)

    durations = []

    for on, off in zip(onsets, offsets):
        #print((off - on) / samprate_Hz)
        pulse_duration = (off - on) / samprate_Hz # seconds

        error = abs(pulse_duration - target)
        assert error < epsilon, \
                'unexpected ' + msg + '. target: ' + str(target)[:4] + ' error: ' + str(error)

    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_odorpulse(odor_pulses, odor_pulse_len, samprate_Hz, epsilon=0.05):
    """
    Checks the duration of the Arduino's command to the valve is consistent with what we expect, 
    and with itself across trials, each to within a tolerance.
    """
    print('Odor pulse lengths... ', end='')

    check_onset2offset(odor_pulses, odor_pulse_len, samprate_Hz, epsilon, \
            'odor pulse duration')


def check_acquisition_time(acquisition_trigger, scopeLen, samprate_Hz, epsilon=0.07):
    """
    Checks acquisition time commanded by the Arduino is consistent with what we expect, 
    and with itself across trials, each to within a tolerance.
    """
    print('Acquisition time per presentation... ', end='')

    check_onset2offset(acquisition_trigger, scopeLen, samprate_Hz, epsilon, \
            'acquisition time')


def check_odor_onset(acquisition_trigger, odor_pulses, onset, samprate_Hz, epsilon=0.05):
    """
    Checks time between acquisition start (command, not effective, but command -> effective
    is Thor's responsibility and is pretty consistent) is consistent with what we expect, 
    and with itself across trials, each to within a tolerance.
    """
    print('Odor onsets relative to acquisition start... ', end='')

    acq_onsets, _  = threshold_crossings(acquisition_trigger)
    odor_onsets, _ = threshold_crossings(odor_pulses)

    # maybe relax this under some cases
    assert len(acq_onsets) == len(odor_onsets), 'different number of acquisition triggers ' + \
            'and odor triggers. acquisition:' + str(len(acq_onsets)) + \
            ' odor: ' + str(len(odor_onsets))

    onset_delays = []

    for a_on, o_on in zip(acq_onsets, odor_onsets):

        assert o_on > a_on, 'odor and acquisition onset mismatch'

        onset_delay = (o_on - a_on) / samprate_Hz
        onset_delays.append(onset_delay)

        assert abs(onset_delay - onset) < epsilon, \
                'expected onset of ' + str(onset) + ', but got  ' + str(onset_delay)[:4]
    
    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_iti(odor_pulses, iti, samprate_Hz, epsilon=0.05):
    """
    Checks ITI is consistent with what we expect, and with itself across trials, each 
    to within a tolerance.

    Measured as odor pulse onset-to-onset. Assuming odor pulses are all of correct duration
    (which will be checked separately) this should be good.
    """
    print('ITI... ', end='')

    odor_onsets, _ = threshold_crossings(odor_pulses)

    for d in np.diff(odor_onsets):

        assert abs(d / samprate_Hz - iti) < epsilon, \
                'expected iti of ' + str(iti) + ', but got  ' + str(d / samprate_Hz)[:4]
    
    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_iti_framecounter(frame_counter, iti, scopeLen, samprate_Hz, epsilon=0.3):
    """
    Measures the complement of the scopeLen, rather than the ITI.
    """
    print('Complement of recording duration... ', end='')

    # TODO replace with threshold_crossings? borrow this logic there?
    shifted = frame_counter[1:].values
    truncated = frame_counter[:-1].values

    max_period = np.max(np.diff(np.where(shifted > truncated)))

    assert abs(max_period / samprate_Hz - (iti - scopeLen)) < epsilon, \
            'ITI - scopeLen was unexpected: expected ' + str(iti - scopeLen) + \
            ' got ' + str(max_period / samprate_Hz)

    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def windows_all_same_length(windows):
    """
    Returns True if end_i - start_i is the same for all windows (tuples of integers (start_i, end_i))
    Returns False otherwise.
    """

    return len(set(map(lambda w: w[1] - w[0], windows))) == 1


def fix_uneven_window_lengths(windows):
    """
    Converts all windows to windows of the minimum window length in input,
    aligned to start indices.
    """

    min_num_frames = min(map(lambda w: w[1] - w[0], windows))

    print(bcolors.WARNING + 'WARNING: running fix_uneven_windows. could misalign trials.' \
            + bcolors.ENDC)

    '''
    min_num_frames = windows[0][1] - windows[0][0]

    for w in windows:
        num_frames = w[1] - w[0]
        if num_frames < min_num_frames:
            min_num_frames = num_frames
    '''

    frame_warning = 'can not calculate dF/F for any less than 2 frames. ' + \
        'make sure the correct ThorSync data is placed in the directory containing ' + \
        'the ThorImage data. for some reason, at least one input window likely has ' + \
        'length zero.'

    if min_num_frames < 2:
       raise ValueError(frame_warning)

    return list(map(lambda w: (w[0], w[0] + min_num_frames), windows))


# TODO factor last two args into params dict?
def crop_trailing_frames(movie, df, averaging, samprate_Hz):

    # TODO avoid having to do this...
    if type(movie) is not np.ndarray:
        movie = movie.toarray()

    movie = movie.squeeze()

    tonsets, toffsets = threshold_crossings(df['acquisition_trigger'])
    beyond_trigger_floor = []
    for on, off in zip(tonsets, toffsets):
        beyond_trigger_floor.append(np.sum(np.diff(\
                df['frame_counter'][on:off + samprate_Hz]))//averaging)

    min_num_frames = int(min(beyond_trigger_floor))

    print('original movie shape', movie.shape)

    print('WARNING: cropping to ' + str(min_num_frames) + ' (per trial)!')
    print('throwing out at most', int(max(beyond_trigger_floor) - min_num_frames), 'frames.')

    to_concat = []
    past = 0
    kept_frames = []

    for frames in beyond_trigger_floor:
        '''
        tossed = frames - min_num_frames
        if tossed != 0:
            start = past + min_num_frames
            cropped_frames.append((start, start + tossed))
        '''
        kept_frames.append((past, past + min_num_frames - 1))

        #print(movie[past:past+min_num_frames,:,:].shape)

        to_concat.append(movie[past:past+min_num_frames,:,:])
        past += int(frames)

    new_movie = np.concatenate(to_concat, axis=0)
    print('new movie shape', new_movie.shape)
    return new_movie, kept_frames


def calc_odor_onset(acquisition_trigger, odor_pulses, samprate_Hz):
    """
    Returns mean delay between acqusition trigger going high and valve driver 
    pin being sent high.
    """

    acq_onsets, _  = threshold_crossings(acquisition_trigger)
    odor_onsets, _ = threshold_crossings(odor_pulses)

    onset_delays = []

    for a_on, o_on in zip(acq_onsets, odor_onsets):
        onset_delay = (o_on - a_on) / samprate_Hz
        onset_delays.append(onset_delay)

    return np.mean(onset_delays)


def get_active_region(deltaF, thresh, invert=False, debug=False):
    """
    Returns the largest contour in the delta image.  This should enable glomerulus identification.

    Args:
        deltaF (np.ndarray): uint8 encoded image (all pixels from 0 to 255)
    """

    if not invert:
        thresh_mode = cv2.THRESH_BINARY

    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    #ret, threshd = cv2.threshold(deltaF, thresh, np.max(deltaF), thresh_mode)
    ret, threshd = cv2.threshold(deltaF, thresh, 255, thresh_mode)

    kernel = np.ones((3,3), np.uint8)
    first_expanded = cv2.dilate(threshd, kernel, iterations=1)
    eroded = cv2.erode(first_expanded, kernel, iterations=2)
    # this is largely to get smooth ROIs
    dilated = cv2.dilate(eroded, kernel, iterations=3)

    # it is important not to use an approximation. drawContours doesn't behave well,
    # and won't include all pixels in contour correctly later.
    #print(dilated.dtype)
    #print(dilated.shape)
    img, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:

        if debug:
            #tplt.hist_image(deltaF, title='Histogram of image without detectable contours')
            return None, threshd, dilated, None

        # TODO but we don't really expect to find all glomeruli for which we present the cognate
        # odors...
        else:
            raise RuntimeError('either the data is bad or get_active_region parameters are set ' + \
                'such that no contours are being found')

    areaArray = []
    # returns the biggest contour
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    #first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    #find the nth largest contour [n-1][1], in this case 2
    largestcontour = sorteddata[0][1]

    if debug:
        #print('area of largest contour=', sorteddata[0][0])

        # so that we can overlay colored information about contours for 
        # troubleshooting purposes
        # last axis should be color
        contour_img = np.ones((dilated.shape[0], dilated.shape[1], \
                3)).astype(np.uint8)

        # TODO 
        # why does the channel coming from dilated appear to be different intensities 
        # sometimes? isn't dilated binary? i guess it isn't... from looking at range
        contour_img[:,:,0] = dilated * 150

        #dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        # args: desination, contours, contour id (neg -> draw all), color, thickness
        cv2.drawContours(contour_img, largestcontour, -1, (0,0,255), 3)

        x, y, w, h = cv2.boundingRect(largestcontour)
        # args: destination image, point 1, point 2 (other corner), color, thickness
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0,255,0), 5)

        return largestcontour, threshd, dilated, contour_img

    else:
        return largestcontour


# TODO make work for imagej stuff too
def pixels_in_contour(img, contour):
    """
    Assumes that the dimensions of each plane are the last two dimensions.
    """
    # because squeeze doesn't work on Images...
    # i guess because it can't change # of images in sequence?
    if not type(img) is np.ndarray:
        img = img.toarray()

    img = np.squeeze(img)

    if len(img.shape) == 2:
        mask = np.zeros(img.shape, np.uint8)

        # TODO keep in mind that this is different from displayed contour, if using any
        # thickness to display
        cv2.drawContours(mask, contour, -1, 1, -1)
        return img[np.nonzero(mask)]

    else:
        # TODO test this more rigourously
        return np.array([pixels_in_contour(img[i,:,:], contour) for i in range(img.shape[0])])


def extract(img, contour):
    return np.mean(pixels_in_contour(img, contour), axis=1)


# TODO test
def extract_mask(img, mask):
    nzmask = np.nonzero(mask)
    return np.mean(img[:,nzmask[0],nzmask[1]], axis=1)


def test_pixels_in_contour():

    img = np.zeros((128,128)).astype(np.uint8)

    # generate an asymmetric test image
    s1 = 7
    e1 = 17
    s2 = 11
    e2 = 13
    # 0 < thresh < val
    val = 255
    img[s1:e1,s2:e2] = val

    thresh = 125
    thresh_mode = cv2.THRESH_BINARY
    ret, threshd = cv2.threshold(img, thresh, 255, thresh_mode)

    img, contours, hierarchy = cv2.findContours(threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    # the function we actually want to test
    pixel_values = pixels_in_contour(img, contours[0])

    assert np.sum(pixel_values) == (e1 - s1) * (e2 - s2) * val, 'sum=' + str(np.sum(pixel_values))

    img[s1,s2] = 0

    pixel_values = pixels_in_contour(img, contours[0])
    assert np.sum(pixel_values) == (e1 - s1) * (e2 - s2) * val - val, 'sum wrong after modifying img'

    img = np.zeros((1050, 512, 512))
    print(pixels_in_contour(img, contours[0]))
    print(pixels_in_contour(img, contours[0]).shape)


def singular(fs):
    if type(fs) is frozenset and len(fs) != 1:
        return False
    return True


def glomerulus_contour(img, debug=False):
    uint8_normalized_img = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX)\
            .astype(np.uint8)
    thresh = np.percentile(uint8_normalized_img, 99)

    # if debug is false, it will just return hotspot
    contour, threshd, dilated, contour_img = \
            get_active_region(uint8_normalized_img, thresh=thresh, \
            debug=debug)

    return contour


def xy_motion_score(series):
    """
    Return a score (lower better) correlated with amount of XY motion between frames.
    """
    # TODO test it never decreases for perturbations to an image with itself (+ noise?)
    assert False, 'not implemented'


def print_odor_order(d, params):
    """
    Prints information sufficient to make sense of one of the raw TIFs used in the analysis,
    for sanity checking viewing these files externally.
    """
    print('')
    with open(join(d,'generated_pin2odor.p'), 'rb') as f:
        connections = pickle.load(f)
        pin2odor = dict(map(lambda x: (x[0], x[1]), connections))

    thorsync_file = get_thorsync_metafile(d)
    samprate_Hz = get_thorsync_samprate(thorsync_file)
    imaging_metafile = get_thorimage_metafile(d)

    # TODO refactor load_data so it works
    #df, pins, samprate_Hz = load_2p_syncdata(thorsync_file)
    df = load_thor_hdf5(d, exp_type='imaging')
    pins, odor_pulses = decode_odor_used(df['odor_used'], samprate_Hz)

    tif_name = join(d, split(d)[-1] + '_ChanA.tif')
    movie = td.images.fromtif(tif_name)

    averaging = get_thor_averaging(imaging_metafile)
    movie, kept_frames = crop_trailing_frames(movie, df, averaging, samprate_Hz)
    num_frames = movie.shape[0]

    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    odorsetlist = [frozenset({pin2odor[p] for p in s}) for s in pins]

    actual_fps = get_effective_framerate(imaging_metafile)
    actual_onset = calc_odor_onset(df['acquisition_trigger'], odor_pulses, samprate_Hz)

    print(bcolors.OKBLUE + bcolors.BOLD, end='')
    print(tif_name)
    print('Actual frames per second:', actual_fps)
    print('Delay to onset of odor pulse:', actual_onset, 'seconds (by the ' + \
            str(int(np.ceil(actual_fps * actual_onset))) + 'th frame)')

    print('First and last frames in each trial with a given odor:')
    print(bcolors.ENDC, end='')

    last = ''
    for mix, fs in zip(odorsetlist, kept_frames):

        if mix == last:
            print(',', fs[0], 'to', fs[1], end='')
        
        else:
            if not last == '':
                print('')

            print(set([odors.pair2str(o) for o in mix]), ':', fs[0], 'to', fs[1], end='')
        
        last = mix

    print('')


def process_session(d, data_stores, params, recompute=False):
    """
    Analysis pipeline from filenames to the projected, averaged, delta F / F
    image, for one group of files (one block within one fly, not one fly).

    One block is how long I could record before I had to change the odor vials
    (since I could only fit so many into my manifold).
    I have os far run 3 blocks per fly, with the 3rd containing less odors than the first two.

    This is repeated for all blocks within a fly, and the results are aggregated before plotting.
    """
    print(d)
    projections, rois = data_stores

    # check to see if we have already computed results for
    # this directory
    session_analysis_cache = join(d, 'analysis_cache.p')

    # TODO make flag to force redoing everything / this
    if isfile(session_analysis_cache) and not recompute:
        print('using processed data in cache for analysis!\n')
        with open(session_analysis_cache, 'rb') as f:
            session_projections, session_rois, block_df = pickle.load(f)

        projections[d] = session_projections
        rois[d] = session_rois
        return block_df

    #print(get_thor_notes(imaging_metafile))

    # we should have previously established the information loaded below
    # exists for this directory
    with open(join(d,'generated_pin2odor.p'), 'rb') as f:
        connections = pickle.load(f)
        pin2odor = dict(map(lambda x: (x[0], x[1]), connections))

    with open(join(d,'generated_stimorder.p'), 'rb') as f:
        pinsetlist = pickle.load(f)

    timg_meta = get_thorimage_metafile(d)
    original_fps = get_effective_framerate(timg_meta)

    #max_fps = params['downsample_below_fps']
    if 'downsample_below_fps' in params:
        raise NotImplementedError('downsampling not yet supported')

    tif_name = join(d, split(d)[-1] + '_ChanA.tif')
    movie = td.images.fromtif(tif_name)
    # this functions to squeeze out unit dimension and put number of images
    # in the correct position. kind of a hack.
    movie = td.images.fromarray(movie)

    ts_df = load_thor_hdf5(d, exp_type='imaging')

    averaging = get_thor_averaging(timg_meta)
    samprate_Hz = get_thorsync_samprate(get_thorsync_metafile(d))
    # careful... will have to crop movie same way each time for motion correction to be valid
    # check it by eye
    # TODO test
    movie, _ = crop_trailing_frames(movie, ts_df, averaging, samprate_Hz)

    model = generate_motion_correction(d, movie)
    corrected = model.transform(movie)

    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    odorsetlist = [frozenset({pin2odor[p] for p in s}) for s in pinsetlist]
    stimulus_panel = set(odorsetlist)

    repeats = params['repeats']
    num_trials = len(pinsetlist)
    print(movie.shape[0] / num_trials)
    frames_per_trial = movie.shape[0] // num_trials

    ################# TODO remove me and get thunder working
    onset = params['recording_start_to_odor_s']
    actual_fps = get_effective_framerate(timg_meta)
    frames_before = int(np.floor(onset * actual_fps)) + 1
    def delta_fluorescence(signal, windows):
        deltaFs = []
        frames_before = int(np.floor(onset * actual_fps))
        for w in windows:
            region = signal[w[0]:w[1],:,:] + 0.001
            baseline_F = np.mean(region[:frames_before,:,:], axis=0)
            delta_F_normed = np.zeros(region.shape) * np.nan
            for i in range(delta_F_normed.shape[0]):
                delta_F_normed[i,:,:] = (region[i,:,:] - baseline_F) \
                    / baseline_F
            deltaFs.append(delta_F_normed)
        return deltaFs

    kernel_size = 3
    kernel = (kernel_size, kernel_size)
    # if this is zero, it is calculated from kernel_size, which is what i want
    sigmaX = 0

    for i in range(movie.shape[0]):
        movie[i,:,:] = cv2.GaussianBlur(movie[i,:,:], kernel, sigmaX)

    # neither toblocks nor reshape permit splitting images into different groups
    # fix that? against a design decision?
    # seems toblocks is more appropriate than reshaping
    # because they enforce constraints on dimensions in reshaping
    # TODO thunderize if possible. change thunder if OK.
    groups = (movie[i:i+frames_per_trial,:,:] for i in \
            range(round(movie.shape[0] / frames_per_trial)))

    windows = [(i*frames_per_trial,(i+1)*frames_per_trial) \
            for i in range(round(movie.shape[0] / frames_per_trial))]

    print('some frames used', windows[:3], windows[-3:])
    print('movie shape', movie.shape)
    
    deltaF = delta_fluorescence(movie, windows)

    #################
    """
    # neither toblocks nor reshape permit splitting images into different groups
    # fix that? against a design decision?
    # seems toblocks is more appropriate than reshaping
    # because they enforce constraints on dimensions in reshaping
    # TODO thunderize if possible. change thunder if OK.
    groups = (corrected[i:i+frames_per_trial,:,:] for i in \
            range(round(corrected.shape[0] / frames_per_trial)))

    # might modify in place? unclear from docs
    filtered = (x.gaussian_filter(sigma=3) for x in groups)
    
    # still necessary?
    nonzero = [x.subtract(-0.001) for x in filtered]

    onset = params['recording_start_to_odor_s']
    actual_fps = get_effective_framerate(timg_meta)
    frames_before = int(np.floor(onset * actual_fps)) + 1
    # squeeze?
    prestimulus_baseline = [x[:frames_before,:,:].mean() for x in nonzero]
    # TODO could probably make more functional
    #prestimulus_baseline = (x[:frames_before,:,:].mean() for x in nonzero)

    #print('prestim baseline shape', prestimulus_baseline[0].shape)

    deltaF = [iseq.map(lambda i: i/base) for iseq, base in zip(nonzero, prestimulus_baseline)]
    """
    ####?
    # what about now?

    unique_stimuli = len(stimulus_panel)
    by_stimulus = [deltaF[i*repeats:(i+1)*repeats] for i in range(unique_stimuli)]

    # TODO try not to have these by numpy arrays yet
    # to take advantage of thunder
    #avgdF = [np.mean(np.array([x.toarray() for x in g]), axis=0) for g in by_stimulus]
    avgdF = [np.mean(np.array([x for x in g]), axis=0) for g in by_stimulus]

    # can't do this b/c they are numpy arrays not Images
    #maxdF = [i.max_projection(axis=0) for i in avgdF]
    maxdF = [np.max(i, axis=0) for i in avgdF]
    ####?

    stimperblock = odorsetlist[0::repeats]

    session_projections = dict(zip(stimperblock, maxdF))
    # holds images for debugging later
    session_rois = dict()

    '''
    for stim, proj in session_projections.items():
        if singular(stim):
            odor = tuple(stim)[0]

            if odors.is_private(odor):
                glom = odors.uniquely_activates[odor]
                session_rois[glom] = glomerulus_contour(proj, debug=True)
    '''

    ##########################################################################
    # TODO more idiomatic way to just let the last index be any integer?
    # in case i want to compare across trials with different framerates?
    max_frames = max([nd.shape[0] for nd in deltaF])

    trial_counter = dict()
    for p in pinsetlist:
        fs = frozenset(p)
        if fs in trial_counter:
            trial_counter[fs] += 1
        else:
            trial_counter[fs] = 1

    max_trials = max(trial_counter.values())
    condition, fly_id, block  = get_fly_info(d)
    # TODO get max # of blocks and use that. index them from 1.

    # TODO use date and int for order instead of fly_id to impose ordering?
    # though there is string ordering
    # TODO load information about rearing age and air / stuff stored in external
    # csv to add to dataframe
    # TODO verify_integrity argument
    multi_index = pd.MultiIndex.from_product( \
            [[condition], [fly_id], stimulus_panel, \
            range(max_trials), range(max_frames)], \
            names=['condition', 'fly_id', 'odor', 'trial', 'frame'])

    # TODO handle consistently with other rois / contours
    label2ijroi = get_ijrois(d, maxdF[0].shape)
    #columns = list(session_rois.keys()) + ['manual_' + l \
    #        for l in label2ijroi.keys()]
    columns = ['manual_' + l for l in label2ijroi.keys()]

    # + ['block']

    block_df = pd.DataFrame(index=multi_index, columns=columns)
    ##########################################################################

    '''
    for glom, contour in session_rois.items():
        traces = [extract(dF, contour) for dF in deltaF]
        trial = 0

        for mixture, trace in zip(odorsetlist, traces):
            # flatten will go across rows before going to the next column
            # which should be the behavior we want
            block_df[glom].loc[condition, fly_id, mixture, trial] = trace

            # TODO just put this back in the index?
            #block_df['block'].loc[condition, fly_id, mixture, trial] = block

            trial = (trial + 1) % repeats

        # TODO TODO still check this, just in the appropriate place
        # making sure we have distinct data in each column
        # and have not copied it all from one place
        #assert np.sum(np.isclose(profiles[0,:], profiles[1,:])) < 2
    '''

    # TODO why are these not getting loaded?
    for label, mask in label2ijroi.items():
        traces = [extract_mask(dF, mask) for dF in deltaF]
        trial = 0

        session_rois[label] = mask

        # TODO break out filling df bit into function? or just shorten it?
        for mixture, trace in zip(odorsetlist, traces):
            block_df['manual_' + label].loc[condition, fly_id, mixture, trial] = trace

            trial = (trial + 1) % repeats
        
        # TODO TODO check not same as above
    with open(session_analysis_cache, 'wb') as f:
        pickle.dump((session_projections, session_rois, block_df), f)

    '''
    o1 = odorsetlist[0]
    o2 = odorsetlist[5]
    o1_traces = block_df.loc[condition, fly_id, o1, trial]
    o2_traces = block_df.loc[condition, fly_id, o2, trial]

    print(type(o1_traces))
    print(o1_traces.values.dtype)
    print(o1, o2, o1_traces.shape, o2_traces.shape, \
            pd.isnull(o1_traces).sum(), pd.isnull(o2_traces).sum())

    assert not np.any(np.isclose(o1_traces.values.astype(np.float32), \
            o2_traces.values.astype(np.float32)))
    '''

    projections[d] = session_projections
    rois[d] = session_rois
    return block_df


def process_experiment(exp_dir, substring2condition, params, cargs=None):
    # TODO fix docstring
    """
    Args:
        exp_dir: directory containing directories with: 
            -OME TIFFs collected via ThorImage
            -directories holding ThorSync data
            -pickle files holding stimulus metadata
            -possibly cache files generated by previous runs of this analysis
        
        substring2condition: dict mapping substrings, that are only present in directories
            belonging to some experimental condition you want to group on, to a description
            of that condition, which will be used as the key for the 'condition' index in
            the DataFrame to be returned

        params:
            -dict holding information about stimulus parameters to group the stack appropriately
             and set the scales
            -TODO generate somehow in the future -> save in another pickle file?
    
    Returns:
        projections: nested dicts indexed similarly to df, but without the block or frame 
            (because images are averages and projections, respectively). terminal values 
            are 2d numpy arrays holding images summarizing trials

        rois: nested dicts indexed same as above, but holding OpenCV contours at the bottom level
        # TODO TODO will this indexing work?

        df: pandas DataFrame multindexed by rearing condition, fly_id, session #, glomerulus, 
            odor, presentation # (out of block), frame #. Contains signal binned in ROI
            detected for the specific glomerulus.
    """

    if cargs is None:
        test = False
        print_summary_only = False
        recheck_data = False
        recompute = False

    else:
        # TODO pythonic way to split up dict? keep it?
        test = cargs.test
        print_summary_only = cargs.print_summary_only 
        recheck_data = cargs.recheck_data
        recompute = cargs.recompute

    possible_session_dirs = [join(exp_dir, d) for d in os.listdir(exp_dir)]

    # check we have all the files we were expecting (ThorImage metadata + TIFs, ThorSync data, 
    # description of stimulus)
    good_session_dirs = [d for d in possible_session_dirs if good_sessiondir(d)]

    if test:
        print('only running on first dir because you ran with test flag')
        good_session_dirs = [good_session_dirs[0]]

    '''
    good = ['170213_01c_o1_restart']
    session_dirs = [join(exp_dir, g) for g in good]
    '''
    print(good_session_dirs)

    # TODO if it works to edit arguments in process_session,
    # i could have failed trials just not edit them, and this can get pushed into there
    # TODO movie this into own function (if not doing above, would would be preferable. both?)

    directories_cache = join(exp_dir, '.directories_cache.p')

    # TODO make this compatible with test flag somehow or make test flag imply force recheck
    if isfile(directories_cache) and not recheck_data:
        with open(directories_cache, 'rb') as f:
            session_dirs = pickle.load(f)

    else:
        # filter out sessions with something unexpected about their data
        # (missing frames, too many frames, wrong ITI, wrong odor pulse length, etc)
        session_dirs = [d for d in good_session_dirs if check_consistency(d, params)]

        with open(directories_cache, 'wb') as f:
            pickle.dump(session_dirs, f)

    # TODO
    #reasons = dict()
    #session_dirs = [d for d in good_session_dirs if check_consistency(d, params, fail_store=reasons)]
    # condition -> list of session directory dicts
    projections = dict()
    rois = dict()

    '''
    for c in substring2condition.values():
        projections[c] = []
        rois[c] = []
    '''
    
    df = pd.DataFrame()
    # would handle df the same way, but none of df operations seem to let 
    # me modify it in place very easily
    data_stores = (projections, rois)

    for d in session_dirs:
        if print_summary_only:
            print_odor_order(d, params)
        else:
            session_df = process_session(d, data_stores, params, recompute=recompute)
            df = df.append(session_df)

    print('\nInitially considered:')
    for s in good_session_dirs:
        print(s)

    print('\nMet data standards:')
    for s in session_dirs:
        print(s)
    print('')

    # TODO print reasons remainder have failed
    return projections, rois, df


def ijroi2mask(ijroi, shape):
    poly = [(p[1], p[0]) for p in ijroi]
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img)


def get_ijrois(d, shape):
    rois = dict()
    for f in os.listdir(d):
        if isfile(join(d,f)) and '.roi' in f:
            with open(join(d,f), 'rb') as ijf:
                roi = ijroi.read_roi(ijf)
            rois[f[:-4]] = ijroi2mask(roi, shape)

    return rois


def process_imagej_measurements(name):
    df = pd.read_csv(name)
    # TODO load as df in same format and return for plotting / analysis
    return
