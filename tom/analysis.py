
import h5py
import scipy.io
import numpy as np
import os
import hashlib
import pickle
import thunder as td

import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.plotting import figure, show, output_file

# TODO maybe make samprate_Hz and similar global variables?

def load_thor_hdf5(fname, exp_type='pid'):
    f = h5py.File(fname, 'r')
    if exp_type == 'pid':
        return f['AI']['odor_used'], f['AI']['ionization_detector']
    elif exp_type == 'imaging':
        return f['AI']['odor_used'], f['CI']['frame_counter']
    else:
        assert False, 'exp_type not valid'

def load_data(name, exp_type=None):

    assert not exp_type is None, 'must specify an exp_type'

    save_prefix = '.tmp/'

    if name[-3:] == '.h5':
        samprate_Hz = 30000

        if exp_type == 'pid':
            odors_used_analog, ionization = load_thor_hdf5(name, exp_type)
        elif exp_type == 'imaging':
            odors_used_analog, frame_counter = load_thor_hdf5(name, exp_type)
        else:
            assert False, 'invalid exp_type'

        nick = hashlib.md5(name.encode('utf-8')).hexdigest()[:10]

        if (not os.path.exists(save_prefix + nick + '_odors_pulses.npy') \
                or not os.path.exists(save_prefix + nick + '_pins.p')):

            if not os.path.exists(save_prefix):
                os.mkdir(save_prefix)

            pins, odor_pulses = decode_odor_used(odors_used_analog)
            np.save(save_prefix + nick + '_odors_pulses', odor_pulses)
            with open(save_prefix + nick + '_pins.p', 'wb') as f:
                pickle.dump(pins, f)
        else:
            odor_pulses = np.load(save_prefix + nick + '_odors_pulses.npy')
            with open(save_prefix + nick + '_pins.p', 'rb') as f:
                pins = pickle.load(f)

    elif name[-4:] == '.mat' and exp_type == 'pid':
        samprate_Hz = 200

        data = scipy.io.loadmat(name)['data']

        if data.shape[1] < 2:
            print(name + ' did not appear to have ionization data. Skipping.')
            return
            # TODO ? or is it missing something else? valve? PLOT

        # at 200Hz
        '''
        odor_pulses = data[:int(data[:,1].shape[0] * (1 - discard_post)),1]
        ionization = data[:int(data[:,0].shape[0] * (1 - discard_post)),0]
        '''
        odor_pulses = data[:int(data[:,1].shape[0]),1]
        ionization = data[:int(data[:,0].shape[0]),0]
    
        # there is not something that can vary as on the olfactometer
        pins = None
    else:
        assert False, 'not sure how to load data'

    if exp_type == 'pid':
        return odor_pulses, pins, ionization, samprate_Hz
    elif exp_type == 'imaging':
        return odor_pulses, pins, frame_counter, samprate_Hz

def load_pid_data(name):
    return load_data(name, exp_type='pid')

def load_2p_syncdata(name):
    return load_data(name, exp_type='imaging')

def decode_odor_used(odor_used_analog, samprate_Hz=30000, verbose=True):
    """ At the beginning of every trial, the Arduino will output a number of 1ms wide high pulses
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

# TODO could probably speed up above function by vectorizing similar to below?
# most time above is probably spent just finding these crossings
def onset_windows(trigger, secs_before, secs_after, samprate_Hz=30000, frame_counter=None, \
        threshold=2.5, max_before=None, max_after=None, averaging=15):
    """ Returns tuples of (start, end) indices in trigger that flank each onset
    (a positive voltage threshold crossing) by secs_before and secs_after.

    -length of windows will all be the same (haven't tested border cases, but otherwise OK)
    -trigger should be a numpy array
    -secs_before and secs_after both relative to onset of the pulse.
    """

    '''
    print('samprate_Hz=' + str(samprate_Hz))
    print('secs_before=' + str(secs_before))
    print('secs_after=' + str(secs_after))
    '''

    shifted = trigger[1:]
    truncated = trigger[:-1]
    # argwhere and where seem pretty much equivalent with boolean arrays?
    onsets = np.where(np.logical_and(shifted > threshold, truncated < threshold))[0]

    # checking how consistent the delays between odor onset and nearest frames are
    # WARNING: delays range over about 8 or 9 ms  --> potential problems (though generally faster
    # than we care about)
    '''
    frames = list(map(lambda x: int(frame_counter[x]), onsets))
    # get the index of the last occurence of the previous frame number
    # to compare to index of odor onset
    previous = list(map(lambda f: np.argwhere(np.isclose(frame_counter, f - 1))[-1,0], frames))
    assert len(previous) == len(onsets)
    delays = list(map(lambda x: x[1] - x[0], zip(onsets, previous)))
    print(delays)
    '''

    # TODO TODO TODO so is most variability in whether or not last frame is there? or first?
    # matters with how i line things up in averaging... (could easily be 500 ms difference)

    if not max_before is None and not max_after is None:
        assert secs_before <= max_before, \
                'trying to display more seconds before onset than available'

        assert secs_after <= max_after , 'only recorded ' + str(max_before+max_after) + 's per trial'

    '''
    # asserts that onsets happen in a certain position in trial with a certain
    # periodicity. might introduce too much risk of errors.

    else:
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

    if frame_counter is None:
        return list(map(lambda x: (x - int(round(samprate_Hz * secs_before)), x + \
                int(round(samprate_Hz * secs_after))), onsets))
    else:
        return list(map(lambda x: ( int(round(int(frame_counter[x - \
                int(round(samprate_Hz * secs_before))]) / averaging)), \
                int(round(int(frame_counter[x + \
                int(round(samprate_Hz * secs_after))]) /  averaging)) ), onsets))


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


def plot_imaging(image_series, trigger, secs_before, secs_after):
    '''
        # implies this was an imaging experiment
        # in this case, we want triplets, to calculate the prestimulus average and the evoked average
        else:
            windows = onset_windows(trigger, secs_before, secs_after, samprate_Hz=samprate_Hz)
            print(windows)
    '''
    assert False, 'not implemented'

# TODO somehow refactor to have a function that makes one plots (and maybe returns an array of 
# subplots / returns something that stores the subplots / takes arg w/ # of subplots(?)
# and another function that can use that function to group things
# the former should be able to handle n=1 too
# TODO how to handle need to sort out data from randomly ordered trials then?
def plot_ota_ionization(signal, trigger, secs_before, secs_after, title='PID response', \
        subtitles=None, pins=None, pin2odor=None, samprate_Hz=30000, fname=None):
    """

    subtitles is either None or a list of the titles of grouped experiments
    """

    # TODO check no nans in signal

    # implies that signal, trigger, and pins are iterables of what they would otherwise be
    # (all of the same length)
    # can also assume that the windows will be of the same length, and that the triggering
    # signal will be the same in all of the trials
    if not subtitles is None:
        windows = []
        pin2avg = []

        for s, t, p, sub in zip(signal, trigger, pins, subtitles):
#            print(sub)
            # TODO make windows just its first element after this?
            windows.append(onset_windows(t, secs_before, secs_after, samprate_Hz=samprate_Hz))
            pin2avg.append(odor_triggered_average(s, windows[-1], p))

    else:
        windows = onset_windows(trigger, secs_before, secs_after, samprate_Hz=samprate_Hz)

        # will return a dict, but if pins is None will only have one key (-1)
#        print(pins)
        pin2avg = odor_triggered_average(signal, windows, pins)
        '''
        print(signal.shape)
        for k, v in pin2avg.items():
            print(v.shape)
        '''

    # TODO filter?

    # TODO bootstrapping?

    if not subtitles is None:
        # assumes subtitles is of len > 1
        rows = 2
        cols = int(np.ceil(len(subtitles) / rows))

    # the reason I use these two dicts is because I wanted to accomodate
    # experiments where the pin used on a given trial could be assigned randomly
    # (so not all of one come before all of another)

    # TODO shouldn't pins not be None for the 2p experiments?
    elif not pins is None:
        unique_pins = set(pins)
        unique_odors = set(pin2odor.values())
        pin2sbplt = dict()
        sbplt2pin = dict()

        for i, p in enumerate(unique_pins, start=1):
            pin2sbplt[p] = i
            sbplt2pin[i] = p
    
        rows = min(len(pin2odor.keys()), 2)
        cols = int(np.ceil(len(unique_pins) / rows))
    else:
        rows = 1
        cols = 1

    fig, axarr = plt.subplots(rows, cols, sharex=True, sharey=True)

    # put the name of the datafile in the window header
    if not fname is None:
        fig.canvas.set_window_title(fname)

    # a title above all of the subplots
    if (not pin2odor is None) and (title is None or title == '') and len(unique_odors) == 1:
    # if we only used one odor, add the name of that odor to the title
        odor = unique_odors.pop()
        plt.suptitle('PID response to ' + odor)
        # add it back to not screw things up next time we access this
        unique_odors.add(odor)
    else:
        plt.suptitle(title)

    # x and y labels
    # (I would rather not have to manually set the position of these, but seems like common way
    #  to do it)
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center')
    fig.text(0.04, 0.5, 'PID output voltage', va='center', rotation='vertical')

    # turn off any subplots we wont actually use
    for i in range(row):
        for j in range(cols):
            curr = i*cols + j + 1

            if not subtitles is None:
                if curr > len(subtitles):
                    if cols == 1:
                        axarr[i].axis('off')
                    else:
                        axarr[i,j].axis('off')

            elif not pins is None and not curr in sbplt2pin:
                if cols == 1:
                    axarr[i].axis('off')
                else:
                    axarr[i,j].axis('off')

    # assumes all odor pulses are the same shape within an experiment
    # (for display purposes)
    if not subtitles is None:
        example = windows[0]
        trigger = trigger[0]
    else:
        example = windows

    trigger_in_window = trigger[example[0][0]:example[0][1]]
    times = np.linspace(0, (trigger_in_window.shape[0] - 1) / samprate_Hz, num=trigger_in_window.shape[0])

    '''
    first = min(map(lambda x: x[0], windows))
    last = max(map(lambda x: x[0], windows))
    '''

    # [0,1]
    border = 0.10
    #ymax = np.percentile(signal, 90)

    if subtitles is None:
        ymax = np.max(signal) * (1 + border)
        ymin = np.min(signal) * (1 - border)
    else:
        ymax = max(map(lambda x: np.max(x), signal)) * (1 + border)
        ymin = min(map(lambda x: np.min(x), signal)) * (1 - border)

    # plot the individuals traces
    if not subtitles is None:
        for i in range(rows):
            for j in range(cols):
                curr = i*cols + j

                if curr >= len(subtitles):
                    break

                if cols == 1:
                    ax = axarr[i]
                else:
                    ax = axarr[i,j]

                for w in windows[curr]:
                    ax.plot(times, signal[curr][w[0]:w[1]], alpha=0.6, linewidth=0.3)

                ax.set_ylim((ymin, ymax))
                ax.fill_between(times, ymin, ymax, where=trigger_in_window.flatten() > 2.5, \
                        facecolor='black', alpha=0.1)

                curr_pins = set(pin2avg[curr].keys())
                assert len(curr_pins) == 1, 'havent implemented multiple pins + arbitrary group'

                ax.plot(times, pin2avg[curr][curr_pins.pop()], '-', c='black', alpha=0.6, \
                        linewidth=1.5, label='Mean')

                if curr == 0:
                    ax.legend()

                ax.title.set_text(subtitles[curr])

        return

    elif not pins is None:
        for p, w in zip(pins, windows):
            plt_ind = pin2sbplt[p] - 1

            if cols > 1:
                ax = axarr[int(np.floor(plt_ind / cols))][plt_ind % cols]
            elif type(axarr) == np.ndarray:
                ax = axarr[plt_ind]
            else:
                ax = axarr

            '''
            if w[0] == first:
                label = 'First'
            elif w[0] == last:
                label = 'Last'
            else:
                label = None
            '''

            # TODO have one shared background image across all?
            # TODO what do i actually want to plot? i can't superimpose images...
            if not frame_rate is None:
                ax.imshow(signal[windows[0][1]])

            else:
                ax.plot(times, signal[w[0]:w[1]], alpha=0.6, linewidth=0.3)
    else:
        for w in windows:
            ax = axarr
            ax.plot(times, signal[w[0]:w[1]], alpha=0.6, linewidth=0.3)

    # plot average trace for each pin
    for p, avg in pin2avg.items():

        if not pins is None:
            plt_ind = pin2sbplt[p] - 1

            if cols > 1:
                ax = axarr[int(np.floor(plt_ind / cols))][plt_ind % cols]
            elif type(axarr) == np.ndarray:
                ax = axarr[plt_ind]
            else:
                ax = axarr

            if len(unique_odors) > 1:
                ax.title.set_text(pin2odor[p])
            elif len(unique_pins) > 1:
                ax.title.set_text('Valve ' + str(p))

        else:
            ax = axarr

        # only need to do this for each average (since there is one average per subplot)
        ax.set_ylim((ymin, ymax))

        # shade region in trial where there the valve is open
        # TODO make more consistent with voltage threshold used earlier? (don't hardcode 2.5)
        ax.fill_between(times, ymin, ymax, where=trigger_in_window.flatten() > 2.5, facecolor='black', alpha=0.1)

        #ax.plot(times, avg, '-', c=color)
        print(avg.shape)
        ax.plot(times, avg, '-', c='black', alpha=0.6, linewidth=1.5, label='Mean')
        if p == min(pin2avg.keys()):
            ax.legend()


def process_experiment(name, title, subtitles=None, secs_before=1, secs_after=3, pin2odor=None, \
        discard_pre=0, discard_post=0, imaging_file=None, exp_type=None):

    # exp_type must be set by the wrapper function
    # don't want a default though
    assert not exp_type is None, 'must set the exp_type'

    print(name)

    # they should be expressed as [0,1]
    # if there some is >= 1, we would have nothing to analyze
    assert discard_pre + discard_post < 1, 'would discard everything'

    if not discard_pre == 0:
        print('WARNING: discarding first ' + str(discard_pre) + ' of trial ' + name)
        assert False, 'discard not implemented'

    if not discard_post == 0:
        print('WARNING: discarding last ' + str(discard_pre) + ' of trial ' + name)
        assert False, 'discard not implemented'

    # TODO actually implement the discard

    # allowing name to also be an iterable of names (assumed, if it isn't a string)
    if not type(name) is str:
        names = name

        odor_pulses = []
        pins = []
        if exp_type == 'pid':
            ionization = []
        elif exp_type == 'imaging':
            frame_counter = []

        # might not be able to hold all of this in memory...
        # assumes all sampling rates in a group are the same
        for name in names:
            if exp_type == 'pid':
                curr_odor_pulses, curr_pins, curr_ionization, samprate_Hz = load_pid_data(name)
            elif exp_type == 'imaging':
                curr_odor_pulses, curr_pins, curr_frame_counter, samprate_Hz = load_2p_syncdata(name)

            odor_pulses.append(curr_odor_pulses)
            pins.append(curr_pins)
            if exp_type == 'pid':
                ionization.append(curr_ionization)
            elif exp_type == 'imaging':
                frame_counter.append(curr_frame_counter)

    else:
        if exp_type == 'pid':
            odor_pulses, pins, ionization, samprate_Hz = load_pid_data(name)
        elif exp_type == 'imaging':
                odor_pulses, pins, frame_counter, samprate_Hz = load_2p_syncdata(name)


    if exp_type == 'pid':
        plot_ota_ionization(ionization, odor_pulses, secs_before, secs_after, title, \
                subtitles, pins, pin2odor=pin2odor, samprate_Hz=samprate_Hz, fname=name)

    elif exp_type == 'imaging':
        # TODO ota_signal?
        #imaging_data = np.array(Image.open(imaging_file))
        # hopefully this can distinguish stacks from time series?
        imaging_data = td.images.fromtif(imaging_file).toarray()
        scopeLen = 15 # seconds (from OlfStimDelivery Arduino code)
        onset = 1
        
        # TODO count pins and make sure there is equal occurence of everything to make sure
        # trial finished correctly

        # print('len(pins) = ' + str(len(pins)))

        # TODO TODO is there other reason to think (scopeLen * len(pins)) is how long we image for?
        frame_rate = imaging_data.shape[0] / (scopeLen * len(pins))
        print('estimated frame rate (assuming recording duration)', frame_rate, 'Hz')

        signal = imaging_data
        trigger = odor_pulses
        windows = onset_windows(trigger, secs_before, secs_after, samprate_Hz=samprate_Hz, \
                frame_counter=frame_counter, max_before=onset, max_after=scopeLen - onset)
        
        # TODO so is average a running average? how is it weighted? it looks like I still have
        # 30 fps (it does not seem to be a running average from manual)

        # TODO oh maybe the frame counter is incremented for each frame that contributes to the
        # cumulative average, and not just the cumulative average frame once it is done?

        # TODO TODO TODO if secs_before == max_before, (which it currently is) i would expect
        # the first frame number to be 0 or 1
        # FIX!!!!  / explain

        # TODO TODO are there actually instances where we capture 15 instead of 14 frames?
        # or are all maybe 15? fix if so
        # (also see TODO in onset_windows)
        '''
        print('len(windows)=' + str(len(windows)))
        print(windows)
        print(list(map(lambda x: (x[1] - x[0]), windows)))
        print(list(map(lambda x: (x[1] - x[0]) / frame_rate, windows)))

        mf = np.max(frame_counter)
        print(mf, 'max frame')
        # TODO assert these are sufficiently close. why aren't they more close?
        averaging = 15 # frames
        print(mf / frame_rate / averaging, 'seconds implied (at calculated framerate)')
        print(len(pins) * scopeLen, 'expected number of seconds (# trials * length of trial')
        print('')
        '''

        '''
        # TODO calculate actual scopeLen? why the discrepancy? am i calculating frame_rate 
        # incorrectly?

        shifted = frame_counter[1:].flatten()
        truncated = frame_counter[:-1].flatten()
        frame_period = np.median(np.diff(np.where(shifted > truncated)))
        print(frame_period)
        print(frame_period / samprate_Hz)
        print(1 / (frame_period / samprate_Hz))
        print('')

        min_period = np.min(np.diff(np.where(shifted > truncated)))
        print(np.where(shifted > truncated)[:10])
        print(np.where(shifted > truncated)[-10:])
        print(min_period)
        print(min_period / samprate_Hz)
        print(1 / (min_period / samprate_Hz))
        print('')
        '''

        '''
        shifted = frame_counter[1:].flatten()
        truncated = frame_counter[:-1].flatten()

        # are frame counts zero or one indexed? zero.
        # print('min frame count', np.min(frame_counter))

        # TODO ASSERT EXPECTED ITI IS SIMILAR TO THIS (getting ~ 30s)
        max_period = np.max(np.diff(np.where(shifted > truncated)))
        print(max_period)
        print(max_period / samprate_Hz)
        print(1 / (max_period / samprate_Hz))
        '''

        # TODO mf / framerate seems 15x larger than pins * scopeLen? is averaging only for display
        # purposes? Experiment.xml does seem to reflect a 15 frame averaging though...

        # TODO if precise, get a certain # of frames before that frame, and a certain # of frames
        # after (not clear the # of frames is precise though, check this too)

        # will return a dict, but if pins is None will only have one key (-1)
#        print(pins)

        min_num_frames = windows[0][1] - windows[0][0]

        for w in windows:
            num_frames = w[1] - w[0]
            if num_frames < min_num_frames:
                min_num_frames = num_frames

        # hack to fix misalignment TODO change
        windows = list(map(lambda w: (w[0], w[0] + min_num_frames), windows))

        pin2avg = odor_triggered_average(signal, windows, pins)

        for pin, avg_image_series in pin2avg.items():
            # need to set range?
            print(pin)
            # TODO TODO TODO check that this actually behaves like the mean, along appropriate dims
            #print(avg_image_series.shape)
            # TODO for each odor (done separately for each fly, with a plot showing all odors 
            # done for a fly, and averaged across odors)

            # TODO only average within an odor in region that i can identify with a private odor
            '''
            f = figure()
            f.image(image=[
            '''

        #plot_ota_ionization(imaging_data, odor_pulses, secs_before, secs_after, \
        #        pins=pins, pin2odor=pin2odor, samprate_Hz=samprate_Hz, fname=name, \
        #        frame_rate=frame_rate)


def process_pid(name, title, subtitles=None, secs_before=1, secs_after=3, pin2odor=None, \
        discard_pre=0, discard_post=0):

    process_experiment(name, title, subtitles, secs_before, secs_after, pin2odor, \
            discard_pre, discard_post, exp_type='pid')


def process_2p(name, syncdata, secs_before=1, secs_after=3, pin2odor=None, \
        discard_pre=0, discard_post=0):
    # needs syncdata and tifs matched up a priori

    # TODO load tifs in name and match up with syncdata somehow
    # or somehow figure out which syncdata to use in this function? based just on tif name?

    title = '' #?
    process_experiment(syncdata, title, secs_before=secs_before, secs_after=secs_after, \
            pin2odor=pin2odor, discard_pre=discard_pre, discard_post=discard_post, \
            imaging_file=name, exp_type='imaging')

def fix_names(prefix, s, suffix):
    """ Adds prefixes and suffixes and works with nested hierarchies of iterables of strings. """

    if type(s) is str:
        return prefix + s + suffix
    else:
        return tuple(map(lambda x: fix_names(prefix, x, suffix), s))
