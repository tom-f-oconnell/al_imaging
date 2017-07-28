
from . import util as u
from . import cache

from os.path import join, split, getmtime, isdir, isfile, exists
import sys
import traceback
# TODO factor out this use to util
import pickle

import thunder as td
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go

def check_acquisition_triggers_crude(acquisition_trigger, target, \
        samprate_Hz, minimum_high_time_s=1):

    print('Number of observed triggers (debounced)... ', end='')
    onsets, offsets = u.threshold_crossings(acquisition_trigger)

    deliberate_triggers = 0
    for on, off in zip(onsets, offsets):
        assert off > on

        if (off - on) / samprate_Hz > minimum_high_time_s:
            deliberate_triggers += 1

    assert deliberate_triggers == target, u.bcolors.FAIL + 'did not trigger acquisition as many ' + \
            'times as you thought, or experiment was stopped prematurely\ntarget: ' + str(target) + \
            '\ndetected: ' + str(deliberate_triggers) + u.bcolors.ENDC

    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)
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

    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


def check_frames_per_trigger(acquisition_trigger, frame_counter, averaging, num_frames, samprate_Hz):
    
    print('Checking frames per trigger...')
    tonsets, toffsets = u.threshold_crossings(acquisition_trigger)

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

    assert num_frames == sum(beyond_trigger_floor)
    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


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

    onsets, offsets = u.threshold_crossings(signal)

    durations = []

    for on, off in zip(onsets, offsets):
        #print((off - on) / samprate_Hz)
        pulse_duration = (off - on) / samprate_Hz # seconds

        error = abs(pulse_duration - target)
        assert error < epsilon, \
                'unexpected ' + msg + '. target: ' + str(target)[:4] + ' error: ' + str(error)

    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


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

    acq_onsets, _  = u.threshold_crossings(acquisition_trigger)
    odor_onsets, _ = u.threshold_crossings(odor_pulses)

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
    
    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


def check_iti(odor_pulses, iti, samprate_Hz, epsilon=0.05):
    """
    Checks ITI is consistent with what we expect, and with itself across trials, each 
    to within a tolerance.

    Measured as odor pulse onset-to-onset. Assuming odor pulses are all of correct duration
    (which will be checked separately) this should be good.
    """
    print('ITI... ', end='')

    odor_onsets, _ = u.threshold_crossings(odor_pulses)

    for d in np.diff(odor_onsets):

        assert abs(d / samprate_Hz - iti) < epsilon, \
                'expected iti of ' + str(iti) + ', but got  ' + str(d / samprate_Hz)[:4]
    
    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


def check_iti_framecounter(frame_counter, iti, scopeLen, samprate_Hz, epsilon=0.3):
    """
    Measures the complement of the scopeLen, rather than the ITI.
    """
    print('Complement of recording duration... ', end='')

    # TODO replace with u.threshold_crossings? borrow this logic there?
    shifted = frame_counter[1:].values
    truncated = frame_counter[:-1].values

    max_period = np.max(np.diff(np.where(shifted > truncated)))

    assert abs(max_period / samprate_Hz - (iti - scopeLen)) < epsilon, \
            'ITI - scopeLen was unexpected: expected ' + str(iti - scopeLen) + \
            ' got ' + str(max_period / samprate_Hz)

    print(u.bcolors.OKGREEN + '[OK]' + u.bcolors.ENDC)


def check_consistency(d, stim_params):
    print(u.bcolors.BOLD + u.bcolors.OKBLUE + 'Checking:\n' + d + u.bcolors.ENDC)

    # TODO factor out
    with open(join(d,'generated_pin2odor.p'), 'rb') as f:
        connections = pickle.load(f)
        pin2odor = dict(map(lambda x: (x[0], x[1]), connections))

    with open(join(d,'generated_stimorder.p'), 'rb') as f:
        pins = pickle.load(f)

    imaging_metafile = u.get_thorimage_metafile(d)
    thorsync_metafile = u.get_thorsync_metafile(d)

    actual_fps = u.get_effective_framerate(imaging_metafile)
    averaging = u.get_thor_averaging(imaging_metafile)
    samprate_Hz = u.get_thorsync_samprate(thorsync_metafile)

    # variables from Arduino code determining trial structure
    scopeLen = stim_params['total_recording_s']
    onset = stim_params['recording_start_to_odor_s']
    odor_pulse_len = stim_params['odor_pulse_ms'] / 1000.0

    # this is how i defined it before: onset to onset, not interval between recordings
    # may way to change
    # (scopeLen + time not recording after each trial = 15 + 30 seconds)
    iti = stim_params['ITI_s']

    # TODO refactor?
    arr = td.images.fromtif(join(d, split(d)[-1] + '_ChanA.tif')).toarray().squeeze()
    num_frames = arr.shape[0]

    def err():
        print('\n' + u.bcolors.FAIL, end='')
        traceback.print_exc(file=sys.stdout)
        print('skipping!' + u.bcolors.ENDC + '\n')
        return False

    # TODO i don't actually use the # repeats now i think, but i could
    # TODO do things that don't require decoding before those that do

    num_trials = len(pins)

    try:
        df = u.load_thor_hdf5(d, exp_type='imaging')
        real_pins, odor_pulses = u.decode_odor_used(df['odor_used'], samprate_Hz)

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

    print(u.bcolors.BOLD + u.bcolors.OKGREEN+ 'all checks passed!' + u.bcolors.ENDC + '\n')
    return True


@cache
def consistent_dirs(dirs, params):
    return [d for d in dirs if check_consistency(d, params)]

