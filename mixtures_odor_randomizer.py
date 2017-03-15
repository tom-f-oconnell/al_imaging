
from __future__ import print_function

import os
import random
import datetime
import pickle

import tom.odors
import trial_server

def set_product(s1, s2):
    """
    Returns all pairwise combinations of two sets.
    """
    s = set()
    for e1 in s1:
        for e2 in s2:
            s.add(frozenset({e1, e2}))
    return s

def key2tupleset(dictionary, key):
    # TODO better explanation
    """
    Assumes dictionary values are iterables.
    """
    val = dictionary[key]
    return set(zip(len(val) * [key], val))

def nice_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':', '')

###############################################################################

save_mappings = True
communicate_to_arduino = True

odor_panel = {'2-butanone': (-4, -6, -8),
              'trans-2-hexenal': (-5, -6, -8, -10),
              'pentyl acetate': (-3,),
              'pentanoic acid': (-2, -3, -4),
              'paraffin (mock)': (0,)}

start = ord('A')
manifold_ports = tuple(chr(x) for x in range(start, start + 8))

available_ports = list(manifold_ports)
# will always be the breathing air (not first passed over parafin)
available_ports.remove('A')

# 5 through 11 inclusive
available_pins = tuple(range(5,12))

control = ('paraffin (mock)', 0)
for_pairs_with_others = {('1-hexanol', ?)}
for_all_pairs = {('1-pentanol', ?), ('methyl salicylate', ?), ('geranyl acetate', ?)}

to_present = {frozenset({control})} | set_product(for_all_pairs, for_all_pairs) \
        | set_product(for_pairs_with_others, for_all_pairs)

# for each odor combination we will test
repeats = 1
# TODO check that it was / still is 45
secs_per_repeat = 45  # seconds

all_mappings = []
odors2pins = []
all_stimuli_in_order = []

####################################################################################

# TODO deal with case where can't connect all at once
odors_needed = set()
for s in to_present:
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
to_present_list = list(to_present)
random.shuffle(to_present_list)
expanded = []
for e in to_present_list:
    expanded += [e] * repeats

all_stimuli_in_order.append(expanded)

secs = len(expanded) * secs_per_repeat
m, s = divmod(secs, 60)
print(m, 'minutes', s, 'seconds')
print('')

####################################################################################

total_secs = sum(map(lambda x: len(x), all_stimuli_in_order)) * secs_per_repeat
m, s = divmod(total_secs, 60)
h, m = divmod(m, 60)
print(h, 'hours', m, 'minutes', s, 'seconds')

# TODO compare w/ decoding saved all_stimuli_in_order
# and then possibly skip decoding

# TODO make if not there
output_dir = '../../stimuli/'
if os.path.isdir(output_dir):
    raise IOError('output directory did not exist. make it or fix output_dir variable, and retry.')

if save_mappings:
    filename = output_dir + nice_timestamp() + '.p'

    print(output_dir + filename)
    with open(filename, 'wb') as f:
        pickle.dump((all_mappings, all_stimuli_in_order), f)

    if not os.path.isfile(output_dir + filename):
        raise IOError('file did not exist after saving!!!')

else:
    print('NOT SAVING MAPPINGS!!!')
    
required_pins_in_order = []
for block in range(len(all_stimuli_in_order)):
    order = []
    for mixture in all_stimuli_in_order[block]:
        order.append({odors2pins[block][o] for o in mixture})
    required_pins_in_order.append(order)
    
# so that i can break the communication out into another script if i want to
with open('.pinorder.tmp.p', 'wb') as f:
    pickle.dump((required_pins_in_order, all_mappings), f)

    if not os.path.isfile('.pinorder.tmp.p'):
        raise IOError('.pinorder.tmp.p did not exist after saving!!!')
    
###############################################################

with open('.pinorder.tmp.p', 'rb') as f:
    required_pins_in_order, all_mappings = pickle.load(f)
    
if communicate_to_arduino:
    trial_server.start(required_pins_in_order, port='COM4', mappings=all_mappings)
