# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:05:43 2017

@author: Tom O'Connell
"""

import pickle

stimfile = '../../stimuli/2017-03-18_140403.p'

with open(stimfile, 'rb') as f:
    all_mappings, all_stimuli_in_order = pickle.load(f)
    
odors2pins = []
for i, session in enumerate(all_mappings):
    odors2pins.append(dict())
    for pin, odor, port in session:
        print(pin, '->', odor, '->', port)
        odors2pins[i][odor] = pin

required_pins_in_order = []
for block in range(len(all_stimuli_in_order)):
    order = []
    for mixture in all_stimuli_in_order[block]:
        order.append({odors2pins[block][o] for o in mixture})
    required_pins_in_order.append(order)
    
with open('.pinorder.tmp.p', 'wb') as f:
    pickle.dump((required_pins_in_order, all_mappings), f)
    