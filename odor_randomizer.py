
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

def expand_odors(odor_panel_dict):
    """
    Returns tuple of strings, one representing each odor in odor_panel.
    """
    expanded = []
    for odor, concentrations in odor_panel_dict.items():
        for c in concentrations:
            expanded.append(odor + ' 1e' + str(c))
            
    return tuple(expanded)


def all_used(used_dict):
    return reduce(lambda x, y: x and y, used_dict.values())


def get_unused(used_dict):
    """
    If given a dict with keys -> boolean, will uniformly pick a key from
    the keys that point to False, and return it, setting the boolean for that
    key to True.
    """
    ret = random.choice([p for p, v in used_dict.items() if v == False])
    used_dict[ret] = True
    return ret


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
        

def send_when_asked(ard, odors_to_send):
    """
    Waits for Arduino to send its trial_index, and replies with appropriate
    element of odors_to_send.
    """
    
    while True:
        # TODO reasons to / not to flush?
        ard.flush()
        ard.write()


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
# TODO later. automatically see which are in plane -> present what i want to for each
# of those
# TODO much later -> automatically pick planes based on responses
# could be iterative


###############################################################################
# Format combinations of odors I want to present as elements of a set
###############################################################################

# TODO break this into a function?
broad = ('pentyl acetate', -3)
mock = ('parafin (mock)', 0)
# won't be sensible if multiple compounds are ever mapped to same glom
# in tom.odors.uniquely_activates (but that isn't the case now)
glom2private_name = {v: k[0] for k, v in tom.odors.uniquely_activates.items()}

to_present = set()

for glom in tom.odors.of_interest:
    private_name = glom2private_name[glom]
    
    # need to use frozenset because sets are implemented as hashsets
    # and you can't consistently hash (map to same integer) something 
    # mutable (therefore normal sets cant hold other sets, because 
    # normal sets are mutable)
    private_series = key2tupleset(odor_panel, private_name)
    private_series_sets = {frozenset({t}) for t in private_series}
    # + work for union?
    for_glom = private_series_sets | set_product(private_series, {broad}) | \
        {frozenset({broad})} | {frozenset({mock})}
    to_present = to_present | for_glom

#print_nestedset(to_present)

###############################################################################
# Draw full set in random order, expand each into blocks
###############################################################################

to_present_ordered = list(to_present)
random.shuffle(to_present_ordered)
repeats = 5
expanded = []
for e in to_present_ordered:
    expanded += [e] * 5
print(expanded)

# TODO check that it was 45
secs_per_repeat = 45
total_secs = len(expanded) * secs_per_repeat
m, s = divmod(total_secs, 60)
print(m, 'minutes', s, 'seconds')

# TODO save the above directly as well -> compare w/ decoding
# and then possibly skip decoding

###############################################################################

output_dir = '../../pins2odors/'

start = ord('A')
manifold_ports = tuple(chr(x) for x in range(start, start + 10))

available_ports = list(manifold_ports)
# will always be the breathing air (not first passed over parafin)
available_ports.remove('A')
# not actually too hard to reach
# available_ports.remove('H')

# 5 through 11 inclusive
available_pins = tuple(range(5,12))

odors = expand_odors(odor_panel)
used_odors = {o: False for o in odors}
used_ports = {p: False for p in available_ports}
used_pins = {p: False for p in available_pins}

# TODO sort by manifold port
# and by pin?

# TODO print stopped ports
# TODO only use low pins if < 7 used, or change arduino script to accomodate
# using all pins on < max #

# TODO group by glom
# TODO change what i want to present depending on plane?
# (i think i'll just do more experiments that use the same kind of stim
#  but target different planes across flies)

# to translate odors to pins while communicating with Arduino later
odors2pins = [dict()]
block = 0

all_mappings = []
buffer = []

# need a minimum per plane which includes parafin and pentyl acetate
# (or at least anything included in a set_product w/ something in current
#  plane)
minimum = [tom.odors.pair2str(o) for o in [broad, mock]]
minimum_copy = list(minimum)

# TODO it used to be the case that each odor was only present once (as the 
# middle element of a tuple) in all_mappings, but after adding a minimum set
# of odors, this is no longer true. may break some analysis code. check.

while True:
    if not len(minimum_copy) == 0:
        odor = random.choice(minimum_copy)
        used_odors[odor] = True
        minimum_copy.remove(odor)
    else:
        odor = get_unused(used_odors)
        
    port = get_unused(used_ports)
    pin = get_unused(used_pins)
    
    buffer.append((pin, odor, port))
    # TODO in future just dont convert to string until now
    odors2pins[block][tom.odors.str2pair(odor)] = pin
              
    if all_used(used_odors):
        break
    
    if all_used(used_ports) or all_used(used_pins):
        for pin, odor, port in buffer:
            print(str(pin) + ' -> ' + odor + ' -> ' + str(port))
            
        print('')
        all_mappings.append(buffer)
        buffer = []
        odors2pins.append(dict())
        block += 1
        minimum_copy = minimum
        
        used_ports = {p: False for p in available_ports}
        used_pins = {p: False for p in available_pins}
    
for pin, odor, port in buffer:
    print(str(pin) + ' -> ' + odor + ' -> ' + str(port))
    
all_mappings.append(buffer)
print('')

filename = output_dir + nice_timestamp() + '.p'
print(filename)

if save_mappings:
    with open(filename, 'wb') as f:
        pickle.dump(all_mappings, f)
else:
    print('NOT SAVING MAPPINGS')

# so that i can break the communication out into another script if i want to
with open('.odors2pins.tmp.p', 'wb') as f:
    pickle.dump(odors2pins, f)
    
###############################################################################
    
print(odors2pins)
for b in range(block + 1):
    print([{odors2pins[b][tom.odors.pair2str(o)] for o in e} \
            for e in expanded])

with open('.odors2pins.tmp.p', 'rb') as f:
    odors2pins = pickle.load(f)
    
if communicate_to_arduino:
    port = 'COM3'
    # TODO change timeout?
    ard = serial.Serial(port, 9600, timeout=5)
    send_when_asked(ard, [odors2pins[o] for o in e for e in expanded])
