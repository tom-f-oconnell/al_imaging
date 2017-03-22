
# for backwards compability with Python 2.7
from __future__ import print_function
from __future__ import division

from . import plotting as tplt
from . import odors

import h5py
import scipy.io
import numpy as np
import os
import xml.etree.ElementTree as etree
import re
import hashlib
import pickle
import thunder as td
from registration import CrossCorr
import tifffile
import cv2

import sys
import traceback

import pandas as pd

display_plots = True
# TODO move outside of analysis.
colormap = 'viridis'

check_everything = False
use_thunder_registration = False
spatial_smoothing = True
find_glomeruli = True

''' Quoting the bokeh 0.12.4 documentation:
'Generally, this should be called at the beginning of an interactive session or the top of a script.'
'''
#output_file('.tmp.bokeh.html')

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


def get_fly_id(directory_name):
    # may fail if more underscores in directory name

    return '_'.join(os.path.split(directory_name)[-1].split('_')[:2])


def get_exptime(thorimage_dir):
    expxml = os.path.join(thorimage_dir, 'Experiment.xml')
    return int(etree.parse(expxml).getroot().find('Date').attrib['uTime'])


def get_synctime(thorsync_dir):
    syncxml = os.path.join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return os.path.getmtime(syncxml)


def get_readable_exptime(thorimage_dir):
    expxml = os.path.join(thorimage_dir, 'Experiment.xml')
    return etree.parse(expxml).getroot().find('Date').attrib['date']

# warn if has SyncData in name but fails this?
def is_thorsync_dir(d):
    if not os.path.isdir(d):
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


# TODO maybe remove
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

    full_id = parse_fly_id(filename)

    # contains date fly was run on, and order fly was run in that day
    fly_id = full_id[:-4]
    # last character of middle portion has this information
    condition = full_id.split('_')[1][-1]
    # assumes 1-9 blocks
    block = int(full_id[-1])

    return condition, fly_id, block


def one_d(nd):
    """
    Returns a one dimensional numpy array, for input with only one non-unit length dimension.
    """
    return np.squeeze(np.array(nd))


# TODO more idiomatic way to do this?
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


def load_thor_hdf5(fname, exp_type='pid'):
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
        if (not os.path.exists(save_prefix + nick + '_odors_pulses.npy') \
                or not os.path.exists(save_prefix + nick + '_pins.p')):

            if not os.path.exists(save_prefix):
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


def decode_odor_used(odor_used_analog, samprate_Hz, verbose=True):
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

    if verbose:
        print('decoding odor pins from analog trace...')

    # this is broken out because i was trying to use numba to jit it,
    # and i originally though the hdf5 reader was the problem, but it doesn't
    # seem like it was
    #@jit
    def decode(odor_used_array):
        # to allow for slight errors in timing
        tolerance = 0.05
        pulse_width = 0.001 # sec (1 ms)
        pulse_samples = pulse_width * samprate_Hz

        # need to change if using something like an Arduino Mega with more pins
        # not sure how analog pins would be handled as is? (not that I use them)
        # TODO fix pin 0 or just assert that it can not be used
        # (how to force something to have a compile time error in Arduino?)
        max_pin_number = 13

        # factor of 2 because the Arduino sends the pin low for pulse_width between each high period
        # of duration pulse_width
        max_signaling_samples = int(round(2 * max_pin_number * pulse_samples * (1 + tolerance)))

        # there is some noise right around 5v, possibly also around 0v
        voltage_threshold = 2.5

        # to exclude false positives from very short transients or data acquisition errors
        pulse_samples_min = int(round(pulse_samples * (1 - tolerance)))

        # the same pin will signal when go high (almost exactly--within microseconds of) when the
        # actual valve pin does, so we also need to limit our search to short pulses
        pulse_samples_max = int(round(pulse_samples * (1 + tolerance)))
        total_samples = odor_used_analog.shape[0]

        odor_pins = []

        discard_first_x_samples = 200
        i = discard_first_x_samples

        # counts pulses
        pin = 0
        signal_start = None
        last_negative_crossing = None
        last_positive_crossing = None

        while i < total_samples - 1:

            if i % 100000 == 0 :
                print(str((i * 1.0) / total_samples * 100)[:6] + '%     ', end='\r')

            if odor_used_array[i] < voltage_threshold:
                if odor_used_array[i+1] > voltage_threshold:

                    last_positive_crossing = i

                    if not last_negative_crossing is None and \
                            last_negative_crossing < i - max_signaling_samples:

                        odor_pins.append((pin, signal_start, last_negative_crossing))
                        #print('added ' + str(odor_pins[-1]))
                        
                        '''
                        color = c=np.random.rand(3,1)
                        plt.axvline(x=signal_start, c=color)
                        plt.axvline(x=last_negative_crossing, c=color)
                        '''

                        last_negative_crossing = None
                        pin = 0
                        # we don't yet know if this is a valid pulse
                        # since pin is set to zero, the clause below should fix it
                        signal_start = None

            # we don't actually need to count negative crossings
            # we can just use them to check whether a positive crossing was valid
            elif odor_used_array[i] > voltage_threshold:
                if odor_used_array[i+1] < voltage_threshold:

                    last_negative_crossing = i

                    assert not last_positive_crossing is None, \
                        'observed negative crossing before positive crossing at i=' + str(i)

                    # check whether the last positive crossing encountered was recent enough
                    # to count the intervening period as a signaling pulse (logic high)
                    age = i - last_positive_crossing
                    if age <= pulse_samples_max:

                        if age >= pulse_samples_min:

                            # we just started counting. record start of signalling period.
                            if pin == 0:
                                signal_start = last_positive_crossing

                            pin += 1
                        else:
                            # TODO remove assertion and just filter out if actually happens
                            assert False, 'observed pulse shorter than expected. ' + \
                                    'try a different voltage threshold? i=' + str(i)
                        # else pulse was too short. might have been noise?
                        # TODO how to handle? does this ever happen?

                    else:
                        pin = 0
                        last_negative_crossing = None
                        last_positive_crossing = None
                        signal_start = None

            i += 1

        if verbose:
            print('done.         ')

        return odor_pins

    odor_used_array = np.array(odor_used_analog)

    '''
    plt.figure()
    sub = 15
    plt.plot(np.arange(0,odor_used_array.shape[0],sub), odor_used_array[::sub])
    '''

    pins_and_timing = decode(odor_used_array)
    pins = list(map(lambda x: x[0], pins_and_timing))
    signaling_times = map(lambda x: (x[1], x[2]), pins_and_timing)

    # TODO if can't store all of trace in array at once, will need to return and handle separately
    for e in signaling_times:
        # offset?
        odor_used_array[max(e[0]-1,0):min(e[1]+1,odor_used_array.shape[0] - 1)] = 0

    # TODO test
    counts = dict()
    for p in pins:
        counts[p] = 0
    for p in pins:
        counts[p] = counts[p] + 1
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

    assert np.isclose(num_frames / num_trials, round(num_frames / num_trials)), \
            'num_frames not divisible by num_trials'

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

    ipdb.set_trace()
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


# TODO also attempt to align images to each other?
# (at least within a fly, if i plan to try using same ROIs)
# might just need to treat blocks as having totally different ROIs...
def correct_xy_motion(imaging_data):
    """
    Aligns images within a planar time series to each other.
    """

    print('starting thunder registration...')

    reg = CrossCorr()
    # registering to the middle of the stack, to try and maximize chance of a good registration
    reference = imaging_data[round(imaging_data.shape[0]/2),:,:]

    # TODO TODO TODO maybe pick something in middle of movie to keep things mostly square in case
    # of long drifts?
    registered = np.zeros(imaging_data.shape) * np.nan
    registered[0,:,:] = imaging_data[0,:,:]

    for i in range(1,imaging_data.shape[0]):
        # TODO important to save model if registration library might change implementation
        # without leaving old, but otherwise fine
        # TODO check for small magnitude of transformations?
        model = reg.fit(imaging_data[i,:,:], reference=reference)

        # TODO i would prefer to work with these in their native format if the thunder
        # library admitted a more natural syntax / set of operations
        # TODO ffs how to save this... no library seems to support it
        registered[i,:,:] = model.transform(imaging_data[i,:,:]).toarray()

    if check_everything:
        print('checking for nans...')
        assert np.sum(np.isnan(registered)) == 0, 'nan leftover in thunder registered stack'

    return registered


# TODO
def get_thorsync_samprate(thorsync_metafile):
    assert False, 'not implemented'


def get_thorimage_xml(imaging_metafile):
    tree = etree.parse(imaging_metafile)
    return tree.getroot()


def get_thor_framerate(imaging_metafile):
    """
    Args:
        imaging_metafile: the location of the relevant Experiment.xml file

    Returns:
        the frame rate recorded in imaging_metafile by ThorImageLS.
    """
    root = get_thorimage_xml(imaging_metafile)
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
    root = get_thorimage_xml(imaging_metafile)
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
    root= get_thorimage_xml(imaging_metafile)
    return root.find('ExperimentNotes').attrib['text']


def check_equal_ocurrence(items):
    """
    Asserts each item that occurs in input dictionary does so with the same frequency as other. 
    May suggest experiment did not run properly.

    Args:
        items (dict)
    """

    d = dict()

    for i in items:
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

    # verbose switch?
    if verbose:
        print('estimated frame rate (assuming recording duration)', est_fps, 'Hz')
        print('framerate in metadata', actual_fps, 'Hz')

    assert abs(est_fps - actual_fps) < epsilon, 'expected and actual framerate mismatch'

    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_duration_est(frame_counter, actual_fps, averaging, imaging_data, pins, scopeLen,
        verbose=False, epsilon=0.15):
    """
    Checks we have about as many frames as we'd expect.

    Prints more debugging information with `verbose` keyword argument set to True.
    """

    print('Recording duration... ', end='')

    mf = np.max(frame_counter)

    if verbose:
        print('actual_fps', actual_fps)
        print('averaging', averaging)
        print('actual frames in series', imaging_data.shape[0])
        print('number of odor presentations', len(pins))
        print('target recording duration per trial', scopeLen)
        print('')

        # some of these are redundant with each other
        print(mf / averaging / actual_fps, 'seconds implied (at actual framerate), ' + \
                "assuming all frames in counter contribute to an average (they don't)")
        print(len(pins) * scopeLen, 'expected number of seconds (# trials * length of trial)')
        print(imaging_data.shape[0] / actual_fps, \
                'actual # frames / actual_framerate = actual total recording duration')
        print('')

        print(imaging_data.shape[0] * averaging, \
                'expected max value of frame_counter, given number of frames in the TIF, and ' + \
                'assuming all frames that increment frame_count contribute to an average')
        print(mf, 'actual maximum frame_counter value')
        print('')

        print('the above, divided by number of odor presentations detected')
        print('raw frames per odor presentation')
        print(imaging_data.shape[0] * averaging / len(pins), 'expected')
        print(mf / len(pins), 'actual')
        print('')

        print('effective frames per odor presentation')
        print(imaging_data.shape[0] / len(pins), 'actual')
        print(mf / len(pins) / averaging, 'from frame_counter')
        print('')

        # TODO count raw frames per acquisition trigger high to see if they are all at the 
        # end or something. would hint at how to handle

    # TODO describe magnitude of potential error
    if abs(imaging_data.shape[0] * averaging - mf) >= (actual_fps / 3):
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

    assert abs((imaging_data.shape[0] / len(pins)) - (mf / len(pins) / averaging)) < epsilon, msg
           
    #print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


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
        pulse_duration = (off - on) / samprate_Hz # seconds

        #durations.append(pulse_duration)

        error = abs(pulse_duration - target)
        assert error < epsilon, \
                'unexpected ' + msg + ': ' + str(target)[:4] + ' error: ' + str(error)

    #print(durations)
    
    print(bcolors.OKGREEN + '[OK]' + bcolors.ENDC)


def check_odorpulse(odor_pulses, odor_pulse_len, samprate_Hz, epsilon=0.05):
    """
    Checks the duration of the Arduino's command to the valve is consistent with what we expect, 
    and with itself across trials, each to within a tolerance.
    """
    print('Odor pulse lengths... ', end='')

    check_onset2offset(odor_pulses, odor_pulse_len, samprate_Hz, epsilon, \
            'unexpected odor pulse duration')


def check_acquisition_time(acquisition_trigger, scopeLen, samprate_Hz, epsilon=0.05):
    """
    Checks acquisition time commanded by the Arduino is consistent with what we expect, 
    and with itself across trials, each to within a tolerance.
    """
    print('Acquisition time per presentation... ', end='')

    check_onset2offset(acquisition_trigger, scopeLen, samprate_Hz, epsilon, \
            'unexpected acquisition time')


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


def check_iti_framecounter(frame_counter, iti, scopeLen, samprate_Hz, epsilon=0.16):
    """
    Measures the complement of the scopeLen, rather than the ITI.
    """
    print('Complement of recording duration... ', end='')

    shifted = frame_counter[1:].flatten()
    truncated = frame_counter[:-1].flatten()

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
        #print(frame_warning)
        #print('SKIPPING', name)
       raise ValueError(frame_warning)

    return list(map(lambda w: (w[0], w[0] + min_num_frames), windows))


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


def delta_fluorescence(signal, windows, actual_fps, onset):
    """
    Returns the delta F / F, where the baseline F is the mean of all frames before onset, for
    each window of the signal. Format is a list of 3D ndarrays with dimensions (frame, x, y).

    actual_fps and onset define how many frames it takes until onset, and all frames before
    onset are used for the baseline.
    """

    deltaFs = []
    frames_before = int(np.floor(onset * actual_fps))

    for w in windows:
        # TODO should it be w[1]+1? (I don't think so, but test anyway)
        # the 0.001 is to prevent division by zero errors. a better solution?
        region = signal[w[0]:w[1],:,:] + 0.001

        # could calculate baseline from mode. perhaps over longer window than this. exclude peak
        # (not doing this for now because i have a suffiently long prestimulus period)
        # TODO + 1 in region slice? check size
        baseline_F = np.mean(region[:frames_before,:,:], axis=0)

        # TODO make sure between this and the baseline all frames are used 
        # unless maybe onset frame is not predictably on one side of the odor onset fence
        # then throw just that frame out?

        delta_F_normed = np.zeros(region.shape) * np.nan

        # TODO check we also reach last frame in avg
        for i in range(delta_F_normed.shape[0]):
            delta_F_normed[i,:,:] = (region[i,:,:] - baseline_F) \
                / baseline_F

        if check_everything:
            assert np.sum(np.isnan(delta_F_normed)) == 0, 'nan in delta F'

        deltaFs.append(delta_F_normed)

    return deltaFs


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

    kernel = np.ones((5,5), np.uint8)
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


# TODO test for stacks
def pixels_in_contour(img, contour):
    """
    Assumes that the dimensions of each plane are the last two dimensions.
    """

    # TODO problems?
    img = np.squeeze(img)

    if img.ndim == 2:
        mask = np.zeros(img.shape, np.uint8)

        # TODO keep in mind that this is different from displayed contour, if using thickness 5 to
        # display
        cv2.drawContours(mask, contour, -1, 1, -1)
        return img[np.nonzero(mask)]

    else:
        # TODO test this more rigourously
        return np.array([pixels_in_contour(img[i,:,:], contour) for i in range(img.shape[0])])

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


def glomerulus_contours(odor_panel, odor2deltaF, debug=False):
    """
    Returns a dict of glomerulus names to contours fit (best-effort) to those glomeruli.
    """

    glom2roi = dict()
    thresh_dict = dict()
    contour_img_dict = dict()

    # TODO need to transform this from odors to glom -> odor, st gloms only defined once
    for odor in filter(odors.is_private, odor_panel):
        img = odor2deltaF[odor]
        #print('img=', img)

        # setting the second arg to None makes it return the result in a new array
        uint8_normalized_img = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX)\
                .astype(np.uint8)
        thresh = np.percentile(uint8_normalized_img, 99)

        # if debug is false, it will just return hotspot
        hotspot, threshd, dilated, contour_img = \
                get_active_region(uint8_normalized_img, thresh=thresh, \
                debug=debug)

        # TODO handle in more principled way
        if hotspot is None:
            continue

        glom = odors.uniquely_activates[odors.str2pair(odor)]
        identifier = odor + ' (private for '+ glom  +')'

        # TODO DEAL WITH CASE WHERE GLOM IS DEFINED MULTIPLE TIMES PER BLOCK
        glom2roi[glom] = hotspot
            
        thresh_dict[identifier] = threshd
        contour_img_dict[identifier] = contour_img

    '''
    if debug and display_plots:
        tplt.plot(thresh_dict, title='Threshold applied to delta image')
        tplt.plot(contour_img_dict, title='Detected contours')
    '''

    return glom2roi, contour_img_dict


def xy_motion_score(series):
    """
    Return a score (lower better) correlated with amount of XY motion between frames.
    """
    # TODO test it never decreases for perturbations to an image with itself (+ noise?)
    assert False, 'not implemented'


def print_odor_order(thorsync_file, pin2odor, imaging_file, trial_duration):
    """
    Prints information sufficient to make sense of one of the raw TIFs used in the analysis,
    for sanity checking viewing these files externally.
    """


    # TODO will it always be in this relative position?
    # refactor this into its own function
    imaging_metafile = '/'.join(thorsync_file.split('/')[:-2]) + '/Experiment.xml'

    df, pins, samprate_Hz = load_2p_syncdata(thorsync_file)
    imaging_data = td.images.fromtif(imaging_file)

    # for the time series data, it seems that the second dimension (in .shape) indexes time
    # when working directly with the Thunder format

    try:
        windows = simple_onset_windows(imaging_data.shape[1], len(pins))

    except AssertionError as err:
        print(bcolors.FAIL, end='')
        traceback.print_exc(file=sys.stdout)
        print('SKIPPING!' + bcolors.ENDC)
        return None, None

    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    odors = []
    for p in pins:
        odors.append(pin2odor[p])

    actual_fps = get_thor_framerate(imaging_metafile)
    actual_onset = calc_odor_onset(df['acquisition_trigger'], df['odor_pulses'], samprate_Hz)

    print(bcolors.OKBLUE + bcolors.BOLD, end='')
    print(imaging_file)
    print('Actual frames per second:', actual_fps)
    print('Delay to onset of odor pulse:', actual_onset, 'seconds (by the ' + \
            str(int(np.ceil(actual_fps * actual_onset))) + 'th frame)')

    print('First and last frames in each trial with a given odor:')
    print(bcolors.ENDC, end='')

    last = ''
    for p in zip(odors, windows):

        if p[0] == last:
            print(',', p[1][0], 'to', p[1][1], end='')
        
        else:
            if not last == '':
                print('')

            print(p[0], ':', p[1][0], 'to', p[1][1], end='')
        
        last = p[0]

    print('')


def process_2p_trial(imaging_metafile, imaging_file, df, pins, pin2odor, \
        secs_before, secs_after, samprate_Hz):
    """
    Analysis pipeline from filenames to the projected, averaged, delta F / F
    image, for one group of files (one block within one fly, not one fly).

    One block is how long I could record before I had to change the odor vials
    (since I could only fit so many into my manifold).
    I have os far run 3 blocks per fly, with the 3rd containing less odors than the first two.

    This is repeated for all blocks within a fly, and the results are aggregated before plotting.
    """

    if not os.path.exists(imaging_file):
        # stop trying to process this experiment
        # TODO i think this code is unreachable? remove?
        print('IMAGING FILE ' + imaging_file + ' DID NOT EXIST')
        return None, None

    print(get_thor_notes(imaging_metafile))

    imaging_data = td.images.fromtif(imaging_file).toarray()

    if use_thunder_registration:

        '''
        if 'reg' in thorsync_file:
            # TODO color / prompt
            print('WARNING: it appears the files provided have already been registered')
        '''

        # TODO check for files first?
        imaging_data = correct_xy_motion(imaging_data)

        # output a TIFF for purposes of evaluating the registration, BUT NOT MORE.
        # WILL LOSE POTENTIALLY USEFUL (?) METADATA

        frags = imaging_file.split('/')
        #mkdir
        output_file = '/'.join(frags[:-1]) + '/thunder_registered/' + frags[-1]
        print('saving TIFF to ' + output_file + '...', end='')

        uint16_normalized = cv2.normalize(imaging_data, None, 0.0, 2**16 - 1, cv2.NORM_MINMAX)\
                .astype(np.uint16)
        #data_out = np.swapaxes(np.expand_dims(uint16_normalized, 0), 1, 3)
        data_out = np.expand_dims(uint16_normalized, 0)

        # from tifffile docstring:
        # "the last dimensions [of data] are assumed to be image depth, height, width, and samples"
        # but ImageJ didn't like that, so i changed to above
        tifffile.imsave(output_file, data_out, imagej=True)

        print(' done.')

    if spatial_smoothing:
        kernel_size = 5
        kernel = (kernel_size, kernel_size)
        # if this is zero, it is calculated from kernel_size, which is what i want
        sigmaX = 0

        for i in range(imaging_data.shape[0]):
            imaging_data[i,:,:] = cv2.GaussianBlur(imaging_data[i,:,:], kernel, sigmaX)

    # TODO take these as input. don't need secs_before and secs_after...
    # maybe want to keep them though?

    # variables from Arduino code determining trial structure
    scopeLen = 15 # seconds (from OlfStimDelivery Arduino code)
    onset = 3
    
    actual_fps = get_thor_framerate(imaging_metafile)
    averaging = get_thor_averaging(imaging_metafile)

    if check_everything:
        try:
            # count pins and make sure there is equal occurence of everything to make sure
            # trial finished correctly
            check_equal_ocurrence(pins)

            # TODO might want to factor this back into below. dont need?
            est_fps = imaging_data.shape[0] / (scopeLen * len(pins))

            # asserts framerate is sufficiently close to expected
            check_framerate_est(actual_fps, est_fps)

            check_duration_est(df['frame_counter'], actual_fps, averaging, imaging_data, pins, scopeLen, \
                    verbose=True)

            check_acquisition_time(df['acquisition_trigger'], scopeLen, samprate_Hz, epsilon=0.05)

            odor_pulse_len = 0.5 # seconds
            check_odorpulse(df['odor_pulses'], odor_pulse_len, samprate_Hz, epsilon=0.05)

            check_odor_onset(df['acquisition_trigger'], df['odor_pulses'], onset, samprate_Hz, epsilon=0.05)

            iti = 45 # seconds
            check_iti(df['odor_pulses'], iti, samprate_Hz, epsilon=0.05)

            check_iti_framecounter(df['frame_counter'], iti, scopeLen, samprate_Hz, epsilon=0.16)

        except AssertionError as err:
            print(bcolors.FAIL, end='')
            traceback.print_exc(file=sys.stdout)
            print('SKIPPING!' + bcolors.ENDC)
            
            # to keep the number of arguments we return consistent whether it fails or not
            # kind of hacky
            # TODO fix
            return None, None

    '''
    max_before = onset
    signal = imaging_data
    trigger = df['odor_pulses']

    # debugging
    suspect_windows = onset_windows(trigger, secs_before, secs_after, samprate_Hz=samprate_Hz, \
            frame_counter=df['frame_counter'], acquisition_trig=df['acquisition_trigger'], max_before=max_before, \
            max_after=scopeLen - onset, actual_fps=actual_fps)
    '''

    try:
        windows = simple_onset_windows(imaging_data.shape[0], len(pins))

    except AssertionError as err:
        print(bcolors.FAIL, end='')
        traceback.print_exc(file=sys.stdout)
        print('SKIPPING!' + bcolors.ENDC)
        # TODO avoid having to do this None, None thing
        return None, None
    '''
    print('')
    print(imaging_data.shape[0])
    print(suspect_windows)
    print(windows)
    '''

    # TODO why are the last ~half of windows doubles of the same number (349,349) for just
    # 170213_01c_o1, despite the trial otherwise looking normally (including, seemingly, the
    # thorsync data...)

    if not windows_all_same_length(windows):

        # hack to fix misalignment TODO change
        try:
            # see docstring for exactly what this does
            windows = fix_uneven_window_lengths(windows)
        except ValueError:
            # stop processing this experiment
            return

    else:
        print(bcolors.OKGREEN + 'windows all same length' + bcolors.ENDC)

    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    odors = [pin2odor[p] for p in pins]

    # returns a list of 3-dimensional (time, x, y) ndarrays, indexed as windows
    # onset and actual_fps are used to determine how many frames to use for baseline
    # baseline is defined as the mean of the frames before odor onset (which is only 2-4 frames now)
    # remaining frames are differenced with the baseline
    deltaFs = delta_fluorescence(imaging_data, windows, actual_fps, onset)

    # average the dF/F image series (which starts right after odor onset)
    # across all presentations of the same odor
    odor_to_avg_deltaF = average_within_odor(deltaFs, odors)
    
    return odor_to_avg_deltaF, deltaFs

def process_experiment(exp_dir, substring2condition, stim_params):
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

        stim_params:
            -dict holding information about stimulus parameters to group the stack appropriately
             and set the scales
            -TODO generate somehow in the future -> save in another pickle file?
    
    Returns:
        df: pandas DataFrame multindexed by rearing condition, fly_id, session #, glomerulus, 
            odor, presentation # (out of block), frame #. Contains signal binned in ROI
            detected for the specific glomerulus.

        projections: nested dicts indexed similarly, but without the block or frame (because images 
            are averages and projections, respectively). terminal values are 2d numpy arrays
            holding images summarizing trials
    """

    # TODO match thorsync stuff

    possible_session_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]
    session_dirs = [d for d in possible_session_dirs if is_thorimage_dir(d)]

    for d in session_dirs:
        correct_xy_motion(d)

    # condition -> list of session directory dicts
    projections = dict()
    for c in substring2condition.values():
        projections[c] = []

    for d in session_dirs:
        session_df, projections = process_session(d)
        projections[get_condition(d, substring2condition)].append(projections)
        df.append(session_df)

    return df, projections


'''
# TODO get rid of defaults. particular secs_before and after
def process_experiment(thorsync_file, title, secs_before, secs_after, pin2odor=None, \
        discard_pre=0, discard_post=0, imaging_file=None, exp_type=None):
    """
    Calculates delta image series, generates plots, attempts glomeruli identification, and 
    generates plots within glomeruli.
    """

    # exp_type must be set by the wrapper function
    # don't want a default though
    assert not exp_type is None, 'must set the exp_type'

    print(bcolors.OKBLUE + bcolors.BOLD + 'Processing:' + bcolors.ENDC + bcolors.OKBLUE)
    # TODO title after above on same line, imaging files output after tifs, maybe p2o after that
    if type(thorsync_file) is str:
         print(thorsync_file)
    else:
        for ts in thorsync_file:
            print(ts)
    print(bcolors.ENDC)

    if exp_type == 'imaging' and pin2odor is None:
        raise ValueError('need a pin2odor mapping for imaging experiments')

    # they should be expressed as [0,1]
    # if there some is >= 1, we would have nothing to analyze
    assert discard_pre + discard_post < 1, 'would discard everything'

    if not discard_pre == 0:
        print('WARNING: discarding first ' + str(discard_pre) + ' of trial ' + thorsync_file)
        assert False, 'discard not implemented'

    if not discard_post == 0:
        print('WARNING: discarding last ' + str(discard_pre) + ' of trial ' + thorsync_file)
        assert False, 'discard not implemented'

    # TODO actually implement the discard

    # allowing name to also be an iterable of names (assumed, if it isn't a string)
    if not type(thorsync_file) is str:
        thorsync_files = thorsync_file

        odor_pulses = []
        pins = []
        if exp_type == 'pid':
            ionization = []
        elif exp_type == 'imaging':
            frame_counter = []

        # might not be able to hold all of this in memory...
        # assumes all sampling rates in a group are the same
        # TODO so this explicitly allows multiple files, but the rest of the function currently
        # can't handle
        # TODO refactor. dont want to keep checking type. make more recursive.
        """
        for thorsync_file in thorsync_files:
            if exp_type == 'pid':
                curr_odor_pulses, curr_pins,curr_ionization,samprate_Hz= load_pid_data(thorsync_file)

            odor_pulses.append(curr_odor_pulses)
            pins.append(curr_pins)
            if exp_type == 'pid':
                ionization.append(curr_ionization)
        """

        odor2deltaF = dict()
        odor_panel = set()

        # indices will be defined by union of all multiindices of component block_dfs
        fly_df = pd.DataFrame()

        for thorsync_file, imaging_file, pin2odor in zip(thorsync_files, imaging_file, pin2odor):
            thorsync_df, pins, samprate_Hz = load_2p_syncdata(thorsync_file)

            # TODO remove this after cleaning up after meeting
            odors = [pin2odor[p] for p in pins]
            #

            # TODO will it always be in this relative position?
            # refactor this into its own function
            imaging_metafile = '/'.join(thorsync_file.split('/')[:-2]) + '/Experiment.xml'

            # returns a dict where each value is the average delta F / F
            # averaged after calculating delta F / F
            # baseline is mean prestimulus period. TODO mode (though maybe unecessary now)
            curr_odor2deltaF, rawdFs = process_2p_trial(imaging_metafile, imaging_file, thorsync_df,\
                    pins, pin2odor, secs_before, secs_after, samprate_Hz)

            # if there was an exception and process_2p_trial returned prematurely
            if curr_odor2deltaF is None:
                return

            # combine the information calculated on current trial
            # with that calculated on past trials of same fly
            for k, d in curr_odor2deltaF.items():

                if k in odor2deltaF:
                    raise ValueError('same odor used in two trials. unexpected.')
                
                odor2deltaF[k] = d

            for odor in pin2odor.values():
                odor_panel.add(odor)

            if not find_glomeruli:
                continue

            # not using odor_panel, because that is cumulative
            curr_odor_panel = set(pin2odor.values())

            # TODO FIX. currently just doing this within blocks to avoid a few problems
            curr_odor_to_max_dF = project_each_value(curr_odor2deltaF, lambda v: np.max(v, axis=0))
            glom2roi, contour_img_dict = \
                    glomerulus_contours(curr_odor_panel, curr_odor_to_max_dF, debug=True)

            condition, fly_id, block  = get_fly_info(thorsync_file)
            # TODO more idiomatic way to just let the last index be any integer?
            # in case i want to compare across trials with different framerates?
            max_frames = max([nd.shape[0] for nd in rawdFs])

            trial_counter = dict()
            for p in pins:
                if p in trial_counter:
                    trial_counter[p] += 1
                else:
                    trial_counter[p] = 1
            max_trials = max(trial_counter.values())

            # TODO get max # of blocks and use that. index them from 1.

            # TODO use date and int for order instead of fly_id to impose ordering?
            # though there is string ordering
            # TODO load information about rearing age and air / stuff stored in external
            # csv to add to dataframe
            # TODO verify_integrity argument
            multi_index = pd.MultiIndex.from_product( \
                    [[condition], [fly_id], curr_odor_panel, \
                    range(max_trials), range(max_frames)], \
                    names=['condition', 'fly_id', 'odor', 'trial', 'frame'])

            # TODO only do for glomeruli we have found contours for
            columns = list(glom2roi.keys()) + ['block']
            block_df = pd.DataFrame(index=multi_index, columns=columns)

            # make traces within each identified glomerulus, for all odors tested
            # TODO as long as the glom is still in ROI at that point
            # TODO test for that!
            for glom, roi in glom2roi.items():
                """
                odors2means = pd.DataFrame(columns=[glom])
                odors2cilower = pd.DataFrame(columns=[glom])
                odors2ciupper = pd.DataFrame(columns=[glom])

                glom_df = pd.DataFrame(index=multi_index, columns=[glom])
                """

                # only need to take portions of trace corresponding to each odor here
                for odor in curr_odor2deltaF:
                    # wasn't in CIs TODO double check everything!
                    """
                    # this is operating on the average deltaF for one odor
                    # and for each pixel
                    mean_profile = pixels_in_contour(curr_odor2deltaF[odor], roi)
                    print(mean_profile.shape)

                    # average across all pixesl
                    # individual pixels belonging to the contour are listed along the 1st axis
                    # the 0th axis is the frame number (wrt window)
                    odors2means[odor] = np.mean(mean_profile, axis=1)
                    """

                    # TODO need (mean across pixels) profile for each trial to calculate CIs!!!!
                    # rawdFs are returned as a list indexed same as pins / odors list
                    profiles = []

                    # iterate over each raw delta F image series that was captured for trials
                    # presenting the current odor, and compress each into a scalar over time
                    for rdf in map(lambda x: x[0],filter(lambda x: x[1] == odor,zip(rawdFs, odors))):

                        profiles.append(np.mean(pixels_in_contour(rdf, roi), axis=1))

                    # TODO make into df and manipulated named cols / indices to reshape
                    # (to ensure correctness)?
                    # dimensions of (num_trials, frames)
                    profiles = np.array(profiles)

                    # making sure we have distinct data in each column
                    # and have not copied it all from one place
                    assert np.sum(np.isclose(profiles[0,:], profiles[1,:])) < 2

                    # flatten will go across rows before going to the next column
                    # which should be the behavior we want
                    # TODO test flatten does what i want
                    block_df[glom].loc[condition, fly_id, odor] = profiles.flatten()
                    block_df['block'].loc[condition, fly_id, odor] = block
                    
                    # i changed the above (commented) mean calculation to this
                    # thinking that would make mean fall within CI, BUT IT DOESN'T see todo
                    #odors2means[odor] = np.mean(profiles, axis=0)
                    """
                    odors2means = df_add_col(odors2means, odor, np.mean(profiles, axis=0))


                    # TODO troubleshoot. mean was out of range.
                    """
                    # trials within units run along the 0th axis
                    ci_lowers, ci_uppers = confidence_intervals(np.stack(profiles), tail=5)
                    odors2cilower[odor] = ci_lowers
                    odors2ciupper[odor] = ci_uppers
                    """

                    # TODO check profiles.shape[0] is N. and check ddof behaves as expected.
                    stderr = np.std(profiles, axis=0, ddof=1) / np.sqrt(profiles.shape[0])
                    odors2cilower = df_add_col(odors2cilower, odor, odors2means[odor] - stderr)
                    odors2ciupper = df_add_col(odors2ciupper, odor, odors2means[odor] + stderr)
                    """
                    # end loop over odors

                """
                tplt.plot(odors2means, emin=odors2cilower, emax=odors2ciupper,\
                        title=title + ', odor panel ' + imaging_file[-5] + ', roi=' + glom, \
                        save=True)
                """
                #tplt.plot(odors2traces, title=title + ', odor panel ' + \
                #        imaging_file[-5] + ', roi=' + glom, save=True)

                # end loop over glomeruli

            fly_df = fly_df.append(block_df)

            # TODO group plots across flies somehow? nested subplots?
            if not len(contour_img_dict) == 0:
                tplt.plot(contour_img_dict, title=title + ', odor panel ' + imaging_file[-5], \
                        save=True)
            else:
                print('No contours found for ' + title)

        # TODO why is cmap not handled for just one image?

        # take the maximum intensity projection of the average dF/F series
        # by changing the second argument, you can use different functions to aggregate data
        # along the time dimension, for instance, lambda v: np.median(v, axis=0)
        # (axis=0 is the time dimension)
        odor_to_max_dF = project_each_value(odor2deltaF, lambda v: np.max(v, axis=0))

        # r is for "raw" strings. MPL recommends it for using latex notation w/ $...$
        # F formatting? need to encode second string?
        if display_plots:
            tplt.plot(odor_to_max_dF, title=r'Fly ' + title, cmap=colormap, save=True)

        # TODO currently doing this within blocks. try to use anatomical stack or something
        # to generalize across blocks. may also just modify blocks to mostly contain
        # the odors i want (but may not always be possible)

        # TODO fix ROI binning functions for trials w/ different frame sizes
        # (if and when i get above TODO to work) (not currently attempting, since all within block)

        print('')
        return fly_df

    else:
        if exp_type == 'pid':
            odor_pulses, pins, ionization, samprate_Hz = load_pid_data(thorsync_file)

        elif exp_type == 'imaging':
                acquisition_trig, odor_pulses, pins, frame_counter, samprate_Hz = \
                        load_2p_syncdata(thorsync_file)

    if exp_type == 'pid':
        plot_ota_ionization(ionization, odor_pulses, secs_before, secs_after, title, \
                subtitles, pins, pin2odor=pin2odor, samprate_Hz=samprate_Hz, fname=thorsync_file)

    elif exp_type == 'imaging':
        assert False, 'this branch no longer supported. fix.'

        odor2deltaF, rawdFs = process_2p_trial(imaging_metafile, imaging_file, df, \
                pins, pin2odor, secs_before, secs_after, samprate_Hz)

        # r is for "raw" strings. MPL recommends it for using latex notation w/ $...$
        if display_plots:
            tplt.plot(odor2deltaF, title=r'$\frac{\Delta{}F}{F}$ for fly ' + title, cmap=colormap)

        # TODO bokeh slider w/ time after onset?
        
        # display next to averages?
        # for each fly, display traces for each odor presented, for odors in each class
        # public, private, inhibitory?
        # TODO mixtures of public and private
        # maybe when i make plot w/ multiple flies, group trials coming from same fly
        # by color?

        # for each glomerulus (especially those identifiable across flies)
        # plot all traces within it (for all odors tested)
        # TODO copy Zhanetta's formatting for easy understanding?

        print('')
'''

def process_pid(name, title, secs_before, secs_after, pin2odor=None, \
        discard_pre=0, discard_post=0):

    process_experiment(name, title, subtitles, secs_before, secs_after, pin2odor, \
            discard_pre, discard_post, exp_type='pid')


def process_2p(name, syncdata, secs_before, secs_after, pin2odor=None, \
        discard_pre=0, discard_post=0):
    # needs syncdata and tifs matched up a priori

    # TODO load tifs in name and match up with syncdata somehow
    # or somehow figure out which syncdata to use in this function? based just on tif name?
    # thorimage database?

    # TODO refactor to be prettier / more recursive
    if type(syncdata) is str:
        title = syncdata.split('/')[5][:-3] #?
    elif type(syncdata) is tuple or type(syncdata) is list:
        title = syncdata[0].split('/')[5][:-3] #?

    return process_experiment(syncdata, title, secs_before=secs_before, secs_after=secs_after, \
            pin2odor=pin2odor, discard_pre=discard_pre, discard_post=discard_post, \
            imaging_file=name, exp_type='imaging')


def fix_names(prefix, s, suffix):
    """ Adds prefixes and suffixes and works with nested hierarchies of iterables of strings. """

    if type(s) is str:
        return prefix + s + suffix
    else:
        return tuple(map(lambda x: fix_names(prefix, x, suffix), s))


def process_imagej_measurements(name):
    df = pd.read_csv(name)
    # TODO load as df in same format and return for plotting / analysis
    return
