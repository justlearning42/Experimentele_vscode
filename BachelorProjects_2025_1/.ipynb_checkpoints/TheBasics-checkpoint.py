#%%
import numpy as np
import pyvisa 
import time
import csv
import matplotlib.pyplot as plt

from IPython.display import display, clear_output

rm = pyvisa.ResourceManager()

k2400_heater = rm.open_resource('GPIB0::2::INSTR')
k2400_sample_current = rm.open_resource('GPIB0::1::INSTR') #good
k2000_pt = rm.open_resource('GPIB0::26::INSTR') #good
k2000_sample_voltage = rm.open_resource('GPIB0::3::INSTR') #good

def get_current(device):
    return float(device.query(':MEASure:curr?').split(',')[1])

def get_voltage(device):
    return float(device.query(':MEASure:voltage?'))

def get_fres(device):
    return float(device.query(':MEASure:FRES?'))

def set_current(device, value):
    device.write(f':sour:current {value}')
    device.write(':outp on')

def set_current_range(device, value):
    device.write(f':sour:curr:range {value}')

def get_current_range(device):
    return device.query(':sour:curr:rang?')