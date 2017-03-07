
from __future__ import print_function

import random
from functools import reduce
import datetime
import pickle
import serial

import tom.odors

def set_product(s1, s2):
    """
    Returns unorderd Cartesian product.
    """
    s = set()
    for e1 in s1:
        for e2 in s2:
            s.add(frozenset({e1, e2}))
    return s

def key2tupleset(dictionary, key):
    """
    Assumes dictionary values are iterables.
    """
    val = dictionary[key]
    return set(zip(len(val) * [key], val))

def nice_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':', '')

def print_nestedset(s):
    """
    Because lots of frozensets as elements of something doesn't print well.
    """
    print('{', end='')
    for fs in s:
        # convert the frozensets to sets for printing
        print(set(fs))
    print('}')

def send_when_asked(ard, pins_in_order):
    """
    Waits for Arduino to send its trial_index, and replies with appropriate
    element of odors_to_send.
    """
    
    # TODO reasons to / not to flush?
    ard.flush()

    expected_trial = 1

    for block in pins_in_order:
        print('starting block', block)
        for mixture in block:
            # TODO catch and retry (so we don't have to restart program / reconnect things)
            while True:
                # TODO if this blocks for long periods of time
                # the arduino could make it to where it needs
                # to set pins high without receiving data!
                line = ard.readline()
                print(line)
                if line != '':
                    break

            trial = int(line)
            assert trial == expected_trial
            print(trial, '', end='')
            expected_trial += 1

            for p in mixture:
                ard.write(str(p))
            ard.write(str(-1))
        print('done with block', block)
        # TODO wait for next block

###############################################################################

save_mappings = False
communicate_to_arduino = True

odor_panel = {'2-butanone': (-4, -6, -8),
              'trans-2-hexenal': (-5, -6, -8, -10),
              'pentyl acetate': (-3,),
              'pentanoic acid': (-2, -3, -4),
              'parafin (mock)': (0,)}

# TODO how many of DL5, DM6, and VM7 can i ever see in the same plane
# at the same time? if one is always excluded -> group odors (also see 
# todo below)

# geranyl acetate, methyl salicitate, and phenyl acetaldehyde were
# all just to be used as landmarks, but they don't seem that necessary.
# iba was just (?) to help identify vm2, but not focusing on that now.
# might still want to use 2-heptanone? or ms to disagnose contam?
# see lab/hong/notes/odor_panel.txt for more reasons for odors used.

# set of frozensets of tuples of form:
# (odor_name (str), exponent_of_conc (v/v) (negative int))
# want to target each glomerulus with its private series, PA, 
# PA + private series, parafin, (and mayyybe inhibitory + PA?)
#odor_panel_differs_by_plane = False
# TODO later. automatically see which are in plane -> present what i want to 
# for each of those
# TODO much later -> automatically pick planes based on responses
# could be iterative

start = ord('A')
manifold_ports = tuple(chr(x) for x in range(start, start + 10))

available_ports = list(manifold_ports)
# will always be the breathing air (not first passed over parafin)
available_ports.remove('A')

# 5 through 11 inclusive
available_pins = tuple(range(5,12))

# TODO break this into a function?
broad = ('pentyl acetate', -3)
mock = ('parafin (mock)', 0)

# won't be sensible if multiple compounds are ever mapped to same glom
# in tom.odors.uniquely_activates (but that isn't the case now)
glom2private_name = {v: k[0] for k, v in tom.odors.uniquely_activates.items()}

# for each odor combination we will test
repeats = 5
# TODO check that it was 45
secs_per_repeat = 45

all_mappings = []
odors2pins = []
all_stimuli_in_order = []

for glom in tom.odors.of_interest:
    print(glom)

    private_name = glom2private_name[glom]
    
    # gather up everything we want to present to a given glomerulus
    private_series = key2tupleset(odor_panel, private_name)
    private_series_sets = {frozenset({t}) for t in private_series}
    set_for_glom = private_series_sets | set_product(private_series, {broad}) | \
        {frozenset({broad})} | {frozenset({mock})}

    # since my odor sets per glomerulus are of size < min(# pins, # ports)
    # i can connect all at once
    odors_needed = set()
    for s in set_for_glom:
        for o in s:
            odors_needed.add(o)
    odors = list(odors_needed)
    random.shuffle(odors)

    # TODO
    # randomly break stimuli into groups fitting into the number of 
    # valves / ports we have on the rig
    # ***if odors are ever to be mixed, need to be connected simultaneously***

    # assign them to random pins / ports
    # needs |pins| <= |odors|
    pins = random.sample(available_pins, len(odors))
    ports = random.sample(available_ports, len(odors))

    for pin, odor_pair, port in sorted(zip(pins, odors, ports), key=lambda x: x[0]):
        odor = tom.odors.pair2str(odor_pair)
        print(str(pin) + ' -> ' + odor + ' -> ' + str(port))

    print('stoppers in ports:')
    for port in sorted(filter(lambda x: x not in ports, available_ports)):
        print(port, '', end='')
    print('')
        
    odors2pins.append(dict(zip(odors, pins)))
    all_mappings.append(list(zip(pins, odors, ports)))

    # now determine order in which to present combinations of the connected
    # odors
    to_present = list(set_for_glom)
    random.shuffle(to_present)
    expanded = []
    for e in to_present:
        expanded += [e] * 5

    all_stimuli_in_order.append(expanded)

    secs = len(expanded) * secs_per_repeat
    m, s = divmod(secs, 60)
    print(m, 'minutes', s, 'seconds')
    print('')

total_secs = sum(map(lambda x: len(x), all_stimuli_in_order)) * secs_per_repeat
m, s = divmod(total_secs, 60)
h, m = divmod(m, 60)
print(h, 'hours', m, 'minutes', s, 'seconds')

# TODO compare w/ decoding saved all_stimuli_in_order
# and then possibly skip decoding

output_dir = '../../pins2odors/'

if save_mappings:
    filename = output_dir + nice_timestamp() + '.p'
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(all_mappings, f)
else:
    print('NOT SAVING MAPPINGS!!!')

# so that i can break the communication out into another script if i want to
with open('.odors2pins.tmp.p', 'wb') as f:
    pickle.dump(odors2pins, f)
    
required_pins_in_order = []
for block in range(len(all_stimuli_in_order)):
    order = []
    for mixture in all_stimuli_in_order[block]:
        order.append({odors2pins[block][o] for o in mixture})
    required_pins_in_order.append(order)

with open('.odors2pins.tmp.p', 'rb') as f:
    odors2pins = pickle.load(f)
    
if communicate_to_arduino:
    port = 'COM6'
    # TODO change timeout?
    with serial.Serial(port, 9600, timeout=5) as ard:
        send_when_asked(ard, required_pins_in_order)

