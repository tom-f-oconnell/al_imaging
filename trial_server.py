# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:35:51 2017

@author: User
"""

import pickle
import serial
import re

def readline(ard):
    ret = b''
    while True:
        line = ard.readline()
        if line != b'':
            ret += line
            '''
            print(line)
            print(line[-1])
            print(line[-1] == 10)
            '''
            # 10 is how newline character is sliced out of
            # binary array (10 is ASCII for \n)
            if line[-1] == 10:
                return ret

def send_when_asked(ard, pins_in_order, mappings=None):
    """
    Waits for Arduino to send its trial_index, and replies with appropriate
    element of odors_to_send.
    """
    
    # TODO reasons to / not to flush?
    ard.flush()
    
    b = 0
    for block in pins_in_order:
        expected_trial = 1
        
        if all_mappings is not None:
            for pin, odor_pair, port in sorted(mappings[b], key=lambda x: x[0]):
                odor = tom.odors.pair2str(odor_pair)
                print(str(pin) + ' -> ' + odor + ' -> ' + str(port))
        
        input('Press Enter to start block.')
        
        print('starting block', b + 1)
        # TODO have arduino check which trial it gets and restart
        # if it gets the wrong value
        ard.write(str('start').encode())
        
        for mixture in block:
            line = readline(ard)
            trial = int(line)
            #print(line)
            #print(trial)
            assert trial == expected_trial
            # TODO why is this line not being printed?
            print(str(trial) + '/' + str(len(block)) + '   ', end='\r')
            expected_trial += 1

            for p in mixture:
                ard.write((str(p) + '\r\n').encode())
            # tells the Arduino we are done sending pin numbers
            ard.write((str(-1) + '\r\n').encode())
            
            #print('sent', mixture)
            line = readline(ard)
            #print(line)
            buffer_contents = set(map(int, re.findall(r'\-?\d+', str(line))))
            #bc_copy = set(buffer_contents)
            assert -1 in buffer_contents or -2 in buffer_contents, \
                'buffer should be terminated with either -1 or -2'
                
            if -1 in buffer_contents:
                buffer_contents.remove(-1)
            if -2 in buffer_contents:
                buffer_contents.remove(-2)
                
            assert set(buffer_contents) == mixture, \
                'buffer does not reflect pins we sent. sent: ' +str(mixture) +\
                '\ngot: ' + str(buffer_contents) + '\nline: ' + str(line)
               
        # the Arduino will still ask for pins
        # as that is currently the only time we send it data
        # (apart from start command)
        # maybe redesign?
        readline(ard)
            
        # tell the Arduino we are done with this block of trials
        # and it should stop and wait for the next start signal
        ard.write((str(-2) + '\r\n').encode())
        print('done with block', b + 1)
        b += 1
            
    print('done!')
    
def start(required_pins_in_order, port='COM10', mappings=None):
    # TODO auto detect correct port
    with serial.Serial(port, 57600, timeout=None) as ard:
        send_when_asked(ard, required_pins_in_order, mappings)

if __name__ == '__main__':
       
    # TODO get most recent file and print that
    '''
    output_dir = '../../stimuli/'
    with open(filename, 'wb') as f:
        pickle.dump((all_mappings, all_stimuli_in_order), f)
    '''
    
    with open('.pinorder.tmp.p', 'rb') as f:
        required_pins_in_order, all_mappings = pickle.load(f)
        
    print(required_pins_in_order)
    
    port = 'COM14'
    start(required_pins_in_order, port=port, mappings=all_mappings)