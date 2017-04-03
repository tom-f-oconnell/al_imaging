
from os.path import getmtime, join, split, exists, isdir, isfile
from os import listdir, mkdir, environ
import h5py
import re
import xml.etree.ElementTree as etree
# TODO remove requirement
import hashlib
import pickle
import ijroi
import numpy as np
import pandas as pd
import thunder as td

from . import odors

valid_fly_ids = re.compile(r'(\d{6}_\d{2}(?:e|c)?_)')
# doesn't actually check for the ../../stimuli/ prefix
# assumes in is in stimuli_base_dir above
stimfile_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{6}\.p)')

# are angle / square brackets special chars?
#pinlist_re = re.compile(r'\[\{(?:\d\d?(?:, )?)+\}(?:, )?\]+')
pinlist_re = re.compile(r'\[(?:\[(?:\{(?:\d\d?(?:, )?)+\}(?:, )?)+\](?:, )?)+\]')
pin_odor_port_re = re.compile(r'(\d\d?) -> (.* 1e-?\d\d?) -> ([A-Z])')


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


# TODO config file?
# TODO argument override?
def get_expdir():
    exp_envvar = 'IMAGING_EXP_DIR'
    if exp_envvar in environ:
        return environ[exp_envvar]
    else:
        return None


def one_d(nd):
    """
    Returns a one dimensional numpy array, for input with only one non-unit length dimension.
    """
    return np.squeeze(np.array(nd))


def new_tiff_name(d, c):
    return join(d, split(d)[-1] + '_' + c + '.tif')


# TODO actually check how tifs are indexed?
def is_anatomical_stack(d):
    return 'anat' in d


def read_old_pin2odor(picklefile):
    with open(picklefile, 'rb') as f:
        pin2odor_list = pickle.load(f)

    # TODO convert to tuple?
    return [[(t[0], odors.str2pair(t[1]), t[2]) for t in l] for l in pin2odor_list]


def get_fly_id(directory_name):
    # may fail if more underscores in directory name

    return '_'.join(split(directory_name)[-1].split('_')[:2])


def get_session_id(directory_name):
    return '_'.join(split(directory_name)[-1].split('_')[2])


def get_exptime(thorimage_dir):
    expxml = join(thorimage_dir, 'Experiment.xml')
    return int(etree.parse(expxml).getroot().find('Date').attrib['uTime'])


def get_synctime(thorsync_dir, time=getmtime):
    syncxml = join(thorsync_dir, 'ThorRealTimeDataSettings.xml')
    return time(syncxml)


def get_readable_exptime(thorimage_dir):
    expxml = join(thorimage_dir, 'Experiment.xml')
    return etree.parse(expxml).getroot().find('Date').attrib['date']


def readable_timestamp(unix_timestamp):
    return datetime.datetime.fromtimestamp(unix_timestamp)


# warn if has SyncData in name but fails this?
def is_thorsync_dir(d):
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

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
    
    files = {f for f in listdir(d)}

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
    return True in {is_thorsync_dir(join(p,c)) for c in listdir(p)}


def contains_stimulus_info(d):
    has_pin2odor = False
    has_stimlist = False
    for f in listdir(d):
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


def get_thorimage_metafile(session_dir):
    return join(session_dir, 'Experiment.xml')


def get_thorsync_metafile(session_dir):
    subdirs = [join(session_dir,d) for d in listdir(session_dir)]
    maybe = [d for d in subdirs if is_thorsync_dir(d)]

    if len(maybe) != 1:
        raise AssertionError('too many possible ThorSync data directories in ' + session_dir + \
                ' to automatically get XML metadata. fix manually.')
    else:
        return join(maybe[0], 'ThorRealTimeDataSettings.xml')


def get_thorsync_hdf5(session_dir):
    subdirs = [join(session_dir,d) for d in listdir(session_dir)]
    maybe = [d for d in subdirs if is_thorsync_dir(d)]

    if len(maybe) != 1:
        raise AssertionError('too many possible ThorSync data directories in ' + session_dir + \
                ' to automatically get XML metadata. fix manually.')
    else:
        d = maybe[0]
        num_h5 = 0
        for f in listdir(d):
            if '.h5' in f:
                num_h5 += 1

        if num_h5 > 1:
            print('WARNING: more than one EpisodeXXX.h5. could have picked wrong one!')

        return join(maybe[0], 'Episode001.h5')


def load_thor_hdf5(fname, exp_type='pid'):
    if isdir(fname):
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

        # TODO change to something else. maybe just use cache decorator
        nick = hashlib.md5(name.encode('utf-8')).hexdigest()[:10]

        # this step seems like it might be taking a lot longer now that i switched to pandas?
        if (not exists(save_prefix + nick + '_odors_pulses.npy') \
                or not exists(save_prefix + nick + '_pins.p')):

            if not exists(save_prefix):
                mkdir(save_prefix)

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

    return df, pins, samprate_Hz


# TODO make work again?
def load_2p_syncdata(name):
    return load_data(name, exp_type='imaging')


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
        # TODO warn if outside these bounds
        min_pin_number = 4
        max_pin_number = 13

        # to exclude false positives from very short transients or data acquisition errors
        pulse_samples_min = int(round(pulse_samples * (1 - tolerance)))

        # the same pin will signal when go high (almost exactly--within microseconds of) when the
        # actual valve pin does, so we also need to limit our search to short pulses
        pulse_samples_max = int(round(pulse_samples * (1 + tolerance)))

        # may need to adjust these, or set a separate tolerance
        # i think i am consistently underestimating them
        between_samples_min = int(round(between_samples * (1 - tolerance)))
        between_samples_max = int(round(between_samples * (1 + tolerance)))

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


def decode_from_session_directory(d):
    thorsync_file = get_thorsync_metafile(d)
    samprate_Hz = get_thorsync_samprate(thorsync_file)

    df = load_thor_hdf5(d, exp_type='imaging')
    pins, _ = decode_odor_used(df['odor_used'], samprate_Hz)
    return pins


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


def singular(fs):
    if type(fs) is frozenset and len(fs) != 1:
        return False
    return True


# TODO maybe move back to analysis?
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
    most_discarded = int(max(beyond_trigger_floor) - min_num_frames)

    print('WARNING: cropping to ' + str(min_num_frames) + ' (per trial)!')
    print('throwing out at most', most_discarded, 'frames.')

    to_concat = []
    past = 0
    kept_frames = []

    for frames in beyond_trigger_floor:
        kept_frames.append((past, past + min_num_frames - 1))
        to_concat.append(movie[past:past+min_num_frames,:,:])
        past += int(frames)

    new_movie = np.concatenate(to_concat, axis=0)
    return new_movie, kept_frames


def get_ijrois(d, shape):
    rois = dict()
    for f in listdir(d):
        if isfile(join(d,f)) and '.roi' in f:
            with open(join(d,f), 'rb') as ijf:
                roi = ijroi.read_roi(ijf)
            rois[f[:-4]] = ijroi2mask(roi, shape)

    return rois


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

    print(d)
    tif_name = new_tiff_name(d, 'ChanA')
    movie = td.images.fromtif(tif_name)
    print(movie)

    averaging = get_thor_averaging(imaging_metafile)
    movie, kept_frames = crop_trailing_frames(movie, df, averaging, samprate_Hz)
    num_frames = movie.shape[0]

    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    try:
        odorsetlist = [frozenset({pin2odor[p] for p in s}) for s in pins]
    except KeyError:
        return

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


