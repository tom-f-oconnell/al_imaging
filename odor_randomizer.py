
import random
from functools import reduce
import datetime
import pickle

def expand_odors(odor_panel_dict):
    expanded = []
    for odor, concentrations in odor_panel_dict.items():
        for c in concentrations:
            expanded.append(odor + ' 1e' + str(c))
            
    return tuple(expanded)

def all_used(used_dict):
    return reduce(lambda x, y: x and y, used_dict.values())

def get_unused(used_dict):
    ''' If given a dict with keys -> boolean, will uniformly pick a key from
    the keys that point to False, and return it, setting the boolean for that
    key to True. '''
    
    ret = random.choice([p for p, v in used_dict.items() if v == False])
    used_dict[ret] = True
    return ret

def nice_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':', '')

# 2-butanone 1e-2 is just for PID measurements

odor_panel = {'2-butanone': (-4, -6, -8),
              'trans-2-hexenal': (-5, -6, -8, -10),
              'pentyl acetate': (-3,),
              '2-heptanone': (-2,),
              'phenyl acetaldehyde': (-4,),
              'isobutyl acetate': (-3,),
              'geranyl acetate': (-5,),
              'pentanoic acid': (-2, -3, -4),
              'parafin (mock)': (0,),
              'methyl salicitate': (-5,)}

# m.s. should maybe have been -4?
# TODO include reasons for all of the above

start = ord('A')
manifold_ports = tuple(chr(x) for x in range(start, start + 10))

available_ports = list(manifold_ports)
# will always be the breathing air (not first passed over parafin)
available_ports.remove('A')
# hard to reach
available_ports.remove('H')

# 5 through 11 inclusive
available_pins = tuple(range(5,12))

odors = expand_odors(odor_panel)
used_odors = {o: False for o in odors}
used_ports = {p: False for p in available_ports}
used_pins = {p: False for p in available_pins}

all_mappings = []
buffer = []

# TODO sort by manifold port
# and by pin?

# TODO print stopped ports
# TODO only use low pins if < 7 used, or change arduino script to accomodate
# using all pins on < max #

while not all_used(used_odors):
    odor = get_unused(used_odors)
    port = get_unused(used_ports)
    pin = get_unused(used_pins)
    
    buffer.append((pin, odor, port))
    
    if all_used(used_ports) or all_used(used_pins):
        for pin, odor, port in buffer:
            print(str(pin) + ' -> ' + odor + ' -> ' + str(port))
        print('')
        all_mappings.append(buffer)
        buffer = []
        
        used_ports = {p: False for p in available_ports}
        used_pins = {p: False for p in available_pins}

# print the remainder set of odors as well
for pin, odor, port in buffer:
    print(str(pin) + ' -> ' + odor + ' -> ' + str(port))
all_mappings.append(buffer)
print('')

filename = nice_timestamp() + '.p'
print(filename)

with open(filename, 'wb') as f:
    pickle.dump(all_mappings, f)

    