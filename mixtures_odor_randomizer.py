
from __future__ import print_function

import os
import random
import datetime
import pickle

import tom.odors
import trial_server

def set_product(s1, s2):
    """
    Returns all pairwise combinations of two sets, where pairs can not be the
    same element twice.
    """
    s = set()
    for e1 in s1:
        for e2 in s2:
            if not e1 == e2:
                s.add(frozenset({e1, e2}))
    return s

def set_format(set_of_tuples):
    """
    Converts a set of tuples to a set of frozensets containing those tuples
    as their sole elements. Format consistent with output of set_product.
    """
    
    s = set()
    for t in set_of_tuples:
        # using frozenset because elements of a set have to be hashable
        # and sets are not hashable, while frozensets are
        s.add(frozenset({t}))
    return s

def nice_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':', '')

###############################################################################

save_stuff = True
communicate_to_arduino = True

start = ord('A')
manifold_ports = tuple(chr(x) for x in range(start, start + 8))

available_ports = list(manifold_ports)
# will always be the breathing air (not first passed over parafin)
available_ports.remove('A')

# 5 through 11 inclusive
available_pins = tuple(range(5,12))

control = ('paraffin', 0)
for_pairs_with_others = {('1-hexanol', -3)}
for_all_pairs = {('1-pentanol', -3), ('methyl salicylate', -3), \
                 ('geranyl acetate', -3)}
all_odors = for_pairs_with_others | for_all_pairs

to_present = {frozenset({control})} | \
        set_product(for_all_pairs, for_all_pairs) | \
        set_product(for_pairs_with_others, for_all_pairs) | \
        set_format(all_odors)
        
# for each odor combination we will test
repeats = 3
# just for purposes of displaying time estimate
# DOES NOT AFFECT INTERVAL SET IN ARDUINO CODE
secs_per_repeat = 120  # seconds
max_secs_per_session = 600

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

# now determine order in which to present combinations of the connected
# odors
to_present_list = list(to_present)
random.shuffle(to_present_list)
expanded = []
for e in to_present_list:
    expanded += [e] * repeats
                
# if don't want block structure, can just shuffle expanded here
repeats_per_session = max_secs_per_session // secs_per_repeat
assert repeats_per_session * secs_per_repeat <= max_secs_per_session

for i in range(len(expanded) // repeats_per_session \
            + (len(expanded) % repeats_per_session != 0)):
    
    # in general these could differ across recording sessions, but as coded now,
    # they will be a copy of the same thing each time
    odors2pins.append(dict(zip(odors, pins)))
    all_mappings.append(list(zip(pins, odors, ports)))
    
    all_stimuli_in_order.append(\
        expanded[i*repeats_per_session:(i+1)*repeats_per_session])

# checking my math
counted_repeats = 0
for session in all_stimuli_in_order:
    counted_repeats += len(session)
assert counted_repeats == len(expanded), 'Tom did some math wrong.'

'''
odors2pins.append(dict(zip(odors, pins)))
all_mappings.append(list(zip(pins, odors, ports)))    
all_stimuli_in_order.append(expanded)
'''

'''
secs = len(expanded) * secs_per_repeat
m, s = divmod(secs, 60)
print(m, 'minutes', s, 'seconds')
print('')
'''

####################################################################################

total_secs = sum(map(lambda x: len(x), all_stimuli_in_order)) * secs_per_repeat
m, s = divmod(total_secs, 60)
h, m = divmod(m, 60)
print(h, 'hours', m, 'minutes', s, 'seconds')
print('')

# TODO compare w/ decoding saved all_stimuli_in_order
# and then possibly skip decoding

if save_stuff:
    output_dir = '../../stimuli/'
    # TODO make if not there
    if not os.path.isdir(output_dir):
        raise IOError('output directory did not exist. ' + \
                      'make it or fix output_dir variable, and retry.')
        
    filename = output_dir + nice_timestamp() + '.p'

    print(filename)
    print('')
    with open(filename, 'wb') as f:
        pickle.dump((all_mappings, all_stimuli_in_order), f)

    if not os.path.isfile(filename):
        raise IOError('file did not exist after saving!!!')

else:
    print('NOT SAVING WHICH PINS WERE CONNECTED TO WHICH ODORS!!!')
    
for group in all_stimuli_in_order:
    for mixture in group:
        for odor in mixture:
            print(odor, end=' ')
        print('')
    
required_pins_in_order = []
for block in range(len(all_stimuli_in_order)):
    order = []
    for mixture in all_stimuli_in_order[block]:
        order.append({odors2pins[block][o] for o in mixture})
    required_pins_in_order.append(order)
    
print(required_pins_in_order)
    
if save_stuff:
    # so that i can break the communication out into another script if i want to
    with open('.pinorder.tmp.p', 'wb') as f:
        pickle.dump((required_pins_in_order, all_mappings), f)
        if not os.path.isfile('.pinorder.tmp.p'):
            raise IOError('.pinorder.tmp.p did not exist after saving!!!')
    
###############################################################

if communicate_to_arduino:
    with open('.pinorder.tmp.p', 'rb') as f:
        required_pins_in_order, all_mappings = pickle.load(f)
    
    trial_server.start(required_pins_in_order, port='COM4', mappings=all_mappings)
