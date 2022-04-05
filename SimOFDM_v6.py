import numpy as np
import scipy.interpolate
import binascii
import sys
from random import randrange
import commpy.channelcoding.convcode as cc
import argparse
import ctypes
import gc
#from guppy import hpy
import math

class OFDMSim:

    # https://ieeexplore.ieee.org/document/4270585
    mappingTable256QAM = {
        (0,0,0,0,0,0,0,0) : (-15-15j)/15,
        (0,0,0,0,0,0,0,1) : (-15-13j)/15,
        (0,0,0,0,0,0,1,1) : (-15-11j)/15,
        (0,0,0,0,0,0,1,0) : (-15-9j)/15,
        (0,0,0,0,0,1,1,0) : (-15-7j)/15,
        (0,0,0,0,0,1,1,1) : (-15-5j)/15,
        (0,0,0,0,0,1,0,1) : (-15-3j)/15,
        (0,0,0,0,0,1,0,0) : (-15-1j)/15,
        (0,0,0,0,1,0,0,0) : (-15+15j)/15,
        (0,0,0,0,1,0,0,1) : (-15+13j)/15,
        (0,0,0,0,1,0,1,1) : (-15+11j)/15,
        (0,0,0,0,1,0,1,0) : (-15+9j)/15,
        (0,0,0,0,1,1,1,0) : (-15+7j)/15,
        (0,0,0,0,1,1,1,1) : (-15+5j)/15,
        (0,0,0,0,1,1,0,1) : (-15+3j)/15,
        (0,0,0,0,1,1,0,0) : (-15+1j)/15,
        (0,0,0,1,0,0,0,0) : (-13-15j)/15,
        (0,0,0,1,0,0,0,1) : (-13-13j)/15,
        (0,0,0,1,0,0,1,1) : (-13-11j)/15,
        (0,0,0,1,0,0,1,0) : (-13-9j)/15,
        (0,0,0,1,0,1,1,0) : (-13-7j)/15,
        (0,0,0,1,0,1,1,1) : (-13-5j)/15,
        (0,0,0,1,0,1,0,1) : (-13-3j)/15,
        (0,0,0,1,0,1,0,0) : (-13-1j)/15,
        (0,0,0,1,1,0,0,0) : (-13+15j)/15,
        (0,0,0,1,1,0,0,1) : (-13+13j)/15,
        (0,0,0,1,1,0,1,1) : (-13+11j)/15,
        (0,0,0,1,1,0,1,0) : (-13+9j)/15,
        (0,0,0,1,1,1,1,0) : (-13+7j)/15,
        (0,0,0,1,1,1,1,1) : (-13+5j)/15,
        (0,0,0,1,1,1,0,1) : (-13+3j)/15,
        (0,0,0,1,1,1,0,0) : (-13+1j)/15,
        (0,0,1,1,0,0,0,0) : (-11-15j)/15,
        (0,0,1,1,0,0,0,1) : (-11-13j)/15,
        (0,0,1,1,0,0,1,1) : (-11-11j)/15,
        (0,0,1,1,0,0,1,0) : (-11-9j)/15,
        (0,0,1,1,0,1,1,0) : (-11-7j)/15,
        (0,0,1,1,0,1,1,1) : (-11-5j)/15,
        (0,0,1,1,0,1,0,1) : (-11-3j)/15,
        (0,0,1,1,0,1,0,0) : (-11-1j)/15,
        (0,0,1,1,1,0,0,0) : (-11+15j)/15,
        (0,0,1,1,1,0,0,1) : (-11+13j)/15,
        (0,0,1,1,1,0,1,1) : (-11+11j)/15,
        (0,0,1,1,1,0,1,0) : (-11+9j)/15,
        (0,0,1,1,1,1,1,0) : (-11+7j)/15,
        (0,0,1,1,1,1,1,1) : (-11+5j)/15,
        (0,0,1,1,1,1,0,1) : (-11+3j)/15,
        (0,0,1,1,1,1,0,0) : (-11+1j)/15,
        (0,0,1,0,0,0,0,0) : (-9-15j)/15,
        (0,0,1,0,0,0,0,1) : (-9-13j)/15,
        (0,0,1,0,0,0,1,1) : (-9-11j)/15,
        (0,0,1,0,0,0,1,0) : (-9-9j)/15,
        (0,0,1,0,0,1,1,0) : (-9-7j)/15,
        (0,0,1,0,0,1,1,1) : (-9-5j)/15,
        (0,0,1,0,0,1,0,1) : (-9-3j)/15,
        (0,0,1,0,0,1,0,0) : (-9-1j)/15,
        (0,0,1,0,1,0,0,0) : (-9+15j)/15,
        (0,0,1,0,1,0,0,1) : (-9+13j)/15,
        (0,0,1,0,1,0,1,1) : (-9+11j)/15,
        (0,0,1,0,1,0,1,0) : (-9+9j)/15,
        (0,0,1,0,1,1,1,0) : (-9+7j)/15,
        (0,0,1,0,1,1,1,1) : (-9+5j)/15,
        (0,0,1,0,1,1,0,1) : (-9+3j)/15,
        (0,0,1,0,1,1,0,0) : (-9+1j)/15,
        (0,1,1,0,0,0,0,0) : (-7-15j)/15,
        (0,1,1,0,0,0,0,1) : (-7-13j)/15,
        (0,1,1,0,0,0,1,1) : (-7-11j)/15,
        (0,1,1,0,0,0,1,0) : (-7-9j)/15,
        (0,1,1,0,0,1,1,0) : (-7-7j)/15,
        (0,1,1,0,0,1,1,1) : (-7-5j)/15,
        (0,1,1,0,0,1,0,1) : (-7-3j)/15,
        (0,1,1,0,0,1,0,0) : (-7-1j)/15,
        (0,1,1,0,1,0,0,0) : (-7+15j)/15,
        (0,1,1,0,1,0,0,1) : (-7+13j)/15,
        (0,1,1,0,1,0,1,1) : (-7+11j)/15,
        (0,1,1,0,1,0,1,0) : (-7+9j)/15,
        (0,1,1,0,1,1,1,0) : (-7+7j)/15,
        (0,1,1,0,1,1,1,1) : (-7+5j)/15,
        (0,1,1,0,1,1,0,1) : (-7+3j)/15,
        (0,1,1,0,1,1,0,0) : (-7+1j)/15,
        (0,1,1,1,0,0,0,0) : (-5-15j)/15,
        (0,1,1,1,0,0,0,1) : (-5-13j)/15,
        (0,1,1,1,0,0,1,1) : (-5-11j)/15,
        (0,1,1,1,0,0,1,0) : (-5-9j)/15,
        (0,1,1,1,0,1,1,0) : (-5-7j)/15,
        (0,1,1,1,0,1,1,1) : (-5-5j)/15,
        (0,1,1,1,0,1,0,1) : (-5-3j)/15,
        (0,1,1,1,0,1,0,0) : (-5-1j)/15,
        (0,1,1,1,1,0,0,0) : (-5+15j)/15,
        (0,1,1,1,1,0,0,1) : (-5+13j)/15,
        (0,1,1,1,1,0,1,1) : (-5+11j)/15,
        (0,1,1,1,1,0,1,0) : (-5+9j)/15,
        (0,1,1,1,1,1,1,0) : (-5+7j)/15,
        (0,1,1,1,1,1,1,1) : (-5+5j)/15,
        (0,1,1,1,1,1,0,1) : (-5+3j)/15,
        (0,1,1,1,1,1,0,0) : (-5+1j)/15,
        (0,1,0,1,0,0,0,0) : (-3-15j)/15,
        (0,1,0,1,0,0,0,1) : (-3-13j)/15,
        (0,1,0,1,0,0,1,1) : (-3-11j)/15,
        (0,1,0,1,0,0,1,0) : (-3-9j)/15,
        (0,1,0,1,0,1,1,0) : (-3-7j)/15,
        (0,1,0,1,0,1,1,1) : (-3-5j)/15,
        (0,1,0,1,0,1,0,1) : (-3-3j)/15,
        (0,1,0,1,0,1,0,0) : (-3-1j)/15,
        (0,1,0,1,1,0,0,0) : (-3+15j)/15,
        (0,1,0,1,1,0,0,1) : (-3+13j)/15,
        (0,1,0,1,1,0,1,1) : (-3+11j)/15,
        (0,1,0,1,1,0,1,0) : (-3+9j)/15,
        (0,1,0,1,1,1,1,0) : (-3+7j)/15,
        (0,1,0,1,1,1,1,1) : (-3+5j)/15,
        (0,1,0,1,1,1,0,1) : (-3+3j)/15,
        (0,1,0,1,1,1,0,0) : (-3+1j)/15,
        (0,1,0,0,0,0,0,0) : (-1-15j)/15,
        (0,1,0,0,0,0,0,1) : (-1-13j)/15,
        (0,1,0,0,0,0,1,1) : (-1-11j)/15,
        (0,1,0,0,0,0,1,0) : (-1-9j)/15,
        (0,1,0,0,0,1,1,0) : (-1-7j)/15,
        (0,1,0,0,0,1,1,1) : (-1-5j)/15,
        (0,1,0,0,0,1,0,1) : (-1-3j)/15,
        (0,1,0,0,0,1,0,0) : (-1-1j)/15,
        (0,1,0,0,1,0,0,0) : (-1+15j)/15,
        (0,1,0,0,1,0,0,1) : (-1+13j)/15,
        (0,1,0,0,1,0,1,1) : (-1+11j)/15,
        (0,1,0,0,1,0,1,0) : (-1+9j)/15,
        (0,1,0,0,1,1,1,0) : (-1+7j)/15,
        (0,1,0,0,1,1,1,1) : (-1+5j)/15,
        (0,1,0,0,1,1,0,1) : (-1+3j)/15,
        (0,1,0,0,1,1,0,0) : (-1+1j)/15,
        (1,1,0,0,0,0,0,0) : ( 1-15j)/15,
        (1,1,0,0,0,0,0,1) : ( 1-13j)/15,
        (1,1,0,0,0,0,1,1) : ( 1-11j)/15,
        (1,1,0,0,0,0,1,0) : ( 1-9j)/15,
        (1,1,0,0,0,1,1,0) : ( 1-7j)/15,
        (1,1,0,0,0,1,1,1) : ( 1-5j)/15,
        (1,1,0,0,0,1,0,1) : ( 1-3j)/15,
        (1,1,0,0,0,1,0,0) : ( 1-1j)/15,
        (1,1,0,0,1,0,0,0) : ( 1+15j)/15,
        (1,1,0,0,1,0,0,1) : ( 1+13j)/15,
        (1,1,0,0,1,0,1,1) : ( 1+11j)/15,
        (1,1,0,0,1,0,1,0) : ( 1+9j)/15,
        (1,1,0,0,1,1,1,0) : ( 1+7j)/15,
        (1,1,0,0,1,1,1,1) : ( 1+5j)/15,
        (1,1,0,0,1,1,0,1) : ( 1+3j)/15,
        (1,1,0,0,1,1,0,0) : ( 1+1j)/15,
        (1,1,0,1,0,0,0,0) : ( 3-15j)/15,
        (1,1,0,1,0,0,0,1) : ( 3-13j)/15,
        (1,1,0,1,0,0,1,1) : ( 3-11j)/15,
        (1,1,0,1,0,0,1,0) : ( 3-9j)/15,
        (1,1,0,1,0,1,1,0) : ( 3-7j)/15,
        (1,1,0,1,0,1,1,1) : ( 3-5j)/15,
        (1,1,0,1,0,1,0,1) : ( 3-3j)/15,
        (1,1,0,1,0,1,0,0) : ( 3-1j)/15,
        (1,1,0,1,1,0,0,0) : ( 3+15j)/15,
        (1,1,0,1,1,0,0,1) : ( 3+13j)/15,
        (1,1,0,1,1,0,1,1) : ( 3+11j)/15,
        (1,1,0,1,1,0,1,0) : ( 3+9j)/15,
        (1,1,0,1,1,1,1,0) : ( 3+7j)/15,
        (1,1,0,1,1,1,1,1) : ( 3+5j)/15,
        (1,1,0,1,1,1,0,1) : ( 3+3j)/15,
        (1,1,0,1,1,1,0,0) : ( 3+1j)/15,
        (1,1,1,1,0,0,0,0) : ( 5-15j)/15,
        (1,1,1,1,0,0,0,1) : ( 5-13j)/15,
        (1,1,1,1,0,0,1,1) : ( 5-11j)/15,
        (1,1,1,1,0,0,1,0) : ( 5-9j)/15,
        (1,1,1,1,0,1,1,0) : ( 5-7j)/15,
        (1,1,1,1,0,1,1,1) : ( 5-5j)/15,
        (1,1,1,1,0,1,0,1) : ( 5-3j)/15,
        (1,1,1,1,0,1,0,0) : ( 5-1j)/15,
        (1,1,1,1,1,0,0,0) : ( 5+15j)/15,
        (1,1,1,1,1,0,0,1) : ( 5+13j)/15,
        (1,1,1,1,1,0,1,1) : ( 5+11j)/15,
        (1,1,1,1,1,0,1,0) : ( 5+9j)/15,
        (1,1,1,1,1,1,1,0) : ( 5+7j)/15,
        (1,1,1,1,1,1,1,1) : ( 5+5j)/15,
        (1,1,1,1,1,1,0,1) : ( 5+3j)/15,
        (1,1,1,1,1,1,0,0) : ( 5+1j)/15,
        (1,1,1,0,0,0,0,0) : ( 7-15j)/15,
        (1,1,1,0,0,0,0,1) : ( 7-13j)/15,
        (1,1,1,0,0,0,1,1) : ( 7-11j)/15,
        (1,1,1,0,0,0,1,0) : ( 7-9j)/15,
        (1,1,1,0,0,1,1,0) : ( 7-7j)/15,
        (1,1,1,0,0,1,1,1) : ( 7-5j)/15,
        (1,1,1,0,0,1,0,1) : ( 7-3j)/15,
        (1,1,1,0,0,1,0,0) : ( 7-1j)/15,
        (1,1,1,0,1,0,0,0) : ( 7+15j)/15,
        (1,1,1,0,1,0,0,1) : ( 7+13j)/15,
        (1,1,1,0,1,0,1,1) : ( 7+11j)/15,
        (1,1,1,0,1,0,1,0) : ( 7+9j)/15,
        (1,1,1,0,1,1,1,0) : ( 7+7j)/15,
        (1,1,1,0,1,1,1,1) : ( 7+5j)/15,
        (1,1,1,0,1,1,0,1) : ( 7+3j)/15,
        (1,1,1,0,1,1,0,0) : ( 7+1j)/15,
        (1,0,1,0,0,0,0,0) : ( 9-15j)/15,
        (1,0,1,0,0,0,0,1) : ( 9-13j)/15,
        (1,0,1,0,0,0,1,1) : ( 9-11j)/15,
        (1,0,1,0,0,0,1,0) : ( 9-9j)/15,
        (1,0,1,0,0,1,1,0) : ( 9-7j)/15,
        (1,0,1,0,0,1,1,1) : ( 9-5j)/15,
        (1,0,1,0,0,1,0,1) : ( 9-3j)/15,
        (1,0,1,0,0,1,0,0) : ( 9-1j)/15,
        (1,0,1,0,1,0,0,0) : ( 9+15j)/15,
        (1,0,1,0,1,0,0,1) : ( 9+13j)/15,
        (1,0,1,0,1,0,1,1) : ( 9+11j)/15,
        (1,0,1,0,1,0,1,0) : ( 9+9j)/15,
        (1,0,1,0,1,1,1,0) : ( 9+7j)/15,
        (1,0,1,0,1,1,1,1) : ( 9+5j)/15,
        (1,0,1,0,1,1,0,1) : ( 9+3j)/15,
        (1,0,1,0,1,1,0,0) : ( 9+1j)/15,
        (1,0,1,1,0,0,0,0) : ( 11-15j)/15,
        (1,0,1,1,0,0,0,1) : ( 11-13j)/15,
        (1,0,1,1,0,0,1,1) : ( 11-11j)/15,
        (1,0,1,1,0,0,1,0) : ( 11-9j)/15,
        (1,0,1,1,0,1,1,0) : ( 11-7j)/15,
        (1,0,1,1,0,1,1,1) : ( 11-5j)/15,
        (1,0,1,1,0,1,0,1) : ( 11-3j)/15,
        (1,0,1,1,0,1,0,0) : ( 11-1j)/15,
        (1,0,1,1,1,0,0,0) : ( 11+15j)/15,
        (1,0,1,1,1,0,0,1) : ( 11+13j)/15,
        (1,0,1,1,1,0,1,1) : ( 11+11j)/15,
        (1,0,1,1,1,0,1,0) : ( 11+9j)/15,
        (1,0,1,1,1,1,1,0) : ( 11+7j)/15,
        (1,0,1,1,1,1,1,1) : ( 11+5j)/15,
        (1,0,1,1,1,1,0,1) : ( 11+3j)/15,
        (1,0,1,1,1,1,0,0) : ( 11+1j)/15,
        (1,0,0,1,0,0,0,0) : ( 13-15j)/15,
        (1,0,0,1,0,0,0,1) : ( 13-13j)/15,
        (1,0,0,1,0,0,1,1) : ( 13-11j)/15,
        (1,0,0,1,0,0,1,0) : ( 13-9j)/15,
        (1,0,0,1,0,1,1,0) : ( 13-7j)/15,
        (1,0,0,1,0,1,1,1) : ( 13-5j)/15,
        (1,0,0,1,0,1,0,1) : ( 13-3j)/15,
        (1,0,0,1,0,1,0,0) : ( 13-1j)/15,
        (1,0,0,1,1,0,0,0) : ( 13+15j)/15,
        (1,0,0,1,1,0,0,1) : ( 13+13j)/15,
        (1,0,0,1,1,0,1,1) : ( 13+11j)/15,
        (1,0,0,1,1,0,1,0) : ( 13+9j)/15,
        (1,0,0,1,1,1,1,0) : ( 13+7j)/15,
        (1,0,0,1,1,1,1,1) : ( 13+5j)/15,
        (1,0,0,1,1,1,0,1) : ( 13+3j)/15,
        (1,0,0,1,1,1,0,0) : ( 13+1j)/15,     
        (1,0,0,0,0,0,0,0) : ( 15-15j)/15,
        (1,0,0,0,0,0,0,1) : ( 15-13j)/15,
        (1,0,0,0,0,0,1,1) : ( 15-11j)/15,
        (1,0,0,0,0,0,1,0) : ( 15-9j)/15,
        (1,0,0,0,0,1,1,0) : ( 15-7j)/15,
        (1,0,0,0,0,1,1,1) : ( 15-5j)/15,
        (1,0,0,0,0,1,0,1) : ( 15-3j)/15,
        (1,0,0,0,0,1,0,0) : ( 15-1j)/15,
        (1,0,0,0,1,0,0,0) : ( 15+15j)/15,
        (1,0,0,0,1,0,0,1) : ( 15+13j)/15,
        (1,0,0,0,1,0,1,1) : ( 15+11j)/15,
        (1,0,0,0,1,0,1,0) : ( 15+9j)/15,
        (1,0,0,0,1,1,1,0) : ( 15+7j)/15,
        (1,0,0,0,1,1,1,1) : ( 15+5j)/15,
        (1,0,0,0,1,1,0,1) : ( 15+3j)/15,
        (1,0,0,0,1,1,0,0) : ( 15+1j)/15,
    }

    mappingTable64QAM = {
        (0,0,0,0,0,0) : (-7-7j)/7,
        (0,0,0,0,0,1) : (-7-5j)/7,
        (0,0,0,0,1,1) : (-7-3j)/7,
        (0,0,0,0,1,0) : (-7-1j)/7,
        (0,0,0,1,0,0) : (-7+7j)/7,
        (0,0,0,1,0,1) : (-7+5j)/7,
        (0,0,0,1,1,1) : (-7+3j)/7,
        (0,0,0,1,1,0) : (-7+1j)/7,
        (0,0,1,0,0,0) : (-5-7j)/7,
        (0,0,1,0,0,1) : (-5-5j)/7,
        (0,0,1,0,1,1) : (-5-3j)/7,
        (0,0,1,0,1,0) : (-5-1j)/7,
        (0,0,1,1,0,0) : (-5+7j)/7,
        (0,0,1,1,0,1) : (-5+5j)/7,
        (0,0,1,1,1,1) : (-5+3j)/7,
        (0,0,1,1,1,0) : (-5+1j)/7,
        (0,1,1,0,0,0) : (-3-7j)/7,
        (0,1,1,0,0,1) : (-3-5j)/7,
        (0,1,1,0,1,1) : (-3-3j)/7,
        (0,1,1,0,1,0) : (-3-1j)/7,
        (0,1,1,1,0,0) : (-3+7j)/7,
        (0,1,1,1,0,1) : (-3+5j)/7,
        (0,1,1,1,1,1) : (-3+3j)/7,
        (0,1,1,1,1,0) : (-3+1j)/7,
        (0,1,0,0,0,0) : (-1-7j)/7,
        (0,1,0,0,1,1) : (-1-3j)/7,
        (0,1,0,0,0,1) : (-1-5j)/7,
        (0,1,0,0,1,0) : (-1-1j)/7,
        (0,1,0,1,0,0) : (-1+7j)/7,
        (0,1,0,1,0,1) : (-1+5j)/7,
        (0,1,0,1,1,1) : (-1+3j)/7,
        (0,1,0,1,1,0) : (-1+1j)/7,
        (1,0,0,0,0,0) : ( 7-7j)/7,
        (1,0,0,0,0,1) : ( 7-5j)/7,
        (1,0,0,0,1,1) : ( 7-3j)/7,
        (1,0,0,0,1,0) : ( 7-1j)/7,
        (1,0,0,1,0,1) : ( 7+5j)/7,
        (1,0,0,1,0,0) : ( 7+7j)/7,
        (1,0,0,1,1,0) : ( 7+1j)/7,
        (1,0,0,1,1,1) : ( 7+3j)/7,
        (1,0,1,0,0,0) : ( 5-7j)/7,
        (1,0,1,0,0,1) : ( 5-5j)/7,
        (1,0,1,0,1,1) : ( 5-3j)/7,
        (1,0,1,0,1,0) : ( 5-1j)/7,
        (1,0,1,1,0,0) : ( 5+7j)/7,
        (1,0,1,1,0,1) : ( 5+5j)/7,
        (1,0,1,1,1,1) : ( 5+3j)/7,
        (1,0,1,1,1,0) : ( 5+1j)/7,
        (1,1,1,0,0,0) : ( 3-7j)/7,
        (1,1,1,0,0,1) : ( 3-5j)/7,
        (1,1,1,0,1,1) : ( 3-3j)/7,
        (1,1,1,0,1,0) : ( 3-1j)/7,
        (1,1,1,1,0,0) : ( 3+7j)/7,
        (1,1,1,1,1,1) : ( 3+3j)/7,
        (1,1,1,1,0,1) : ( 3+5j)/7,
        (1,1,1,1,1,0) : ( 3+1j)/7,
        (1,1,0,0,0,0) : ( 1-7j)/7,
        (1,1,0,0,0,1) : ( 1-5j)/7,
        (1,1,0,0,1,1) : ( 1-3j)/7,
        (1,1,0,0,1,0) : ( 1-1j)/7,
        (1,1,0,1,0,0) : ( 1+7j)/7,
        (1,1,0,1,0,1) : ( 1+5j)/7,
        (1,1,0,1,1,1) : ( 1+3j)/7,
        (1,1,0,1,1,0) : ( 1+1j)/7,
    }

    mappingTable16QAM = {
        (0,0,0,0) : (-3-3j)/3,
        (0,0,0,1) : (-3-1j)/3,
        (0,0,1,0) : (-3+3j)/3,
        (0,0,1,1) : (-3+1j)/3,
        (0,1,0,0) : (-1-3j)/3,
        (0,1,0,1) : (-1-1j)/3,
        (0,1,1,0) : (-1+3j)/3,
        (0,1,1,1) : (-1+1j)/3,
        (1,0,0,0) : ( 3-3j)/3,
        (1,0,0,1) : ( 3-1j)/3,
        (1,0,1,0) : ( 3+3j)/3,
        (1,0,1,1) : ( 3+1j)/3,
        (1,1,0,0) : ( 1-3j)/3,
        (1,1,0,1) : ( 1-1j)/3,
        (1,1,1,0) : ( 1+3j)/3,
        (1,1,1,1) : ( 1+1j)/3
    }

    mappingTable8PSK = {
        (0,0,0) : -np.sqrt(2)-np.sqrt(2)*1j,
        (0,0,1) : -1+0j,
        (0,1,0) :  0+1j,
        (0,1,1) : -np.sqrt(2)+np.sqrt(2)*1j,
        (1,0,0) :  0-1j,
        (1,0,1) : +np.sqrt(2)-np.sqrt(2)*1j,
        (1,1,0) : +np.sqrt(2)+np.sqrt(2)*1j,
        (1,1,1) :  1+0j,
    }

    mappingTableQPSK = {
        (0,0) : -np.sqrt(2)-np.sqrt(2)*1j,
        (0,1) : -np.sqrt(2)+np.sqrt(2)*1j,
        (1,0) : +np.sqrt(2)+np.sqrt(2)*1j,
        (1,1) : +np.sqrt(2)-np.sqrt(2)*1j,
    }

    mappingTableBPSK = {
        (0,) : -1,
        (1,) :  1,
    }

    crc4Poly = [1, 0, 0, 1, 1]

    coding12PunctureMatrix = np.array([1])

    coding23PunctureMatrix = np.array([
        [1, 1],
        [1, 0]
    ])

    coding34PunctureMatrix = np.array([
        [1, 1, 0],
        [1, 0, 1]
    ])
    
    # https://stackoverflow.com/questions/54946638/punctured-convolutional-codes-in-gnu-radio
    # https://en.wikipedia.org/wiki/Convolutional_code
    coding56PunctureMatrix = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ])
    
    coding78PunctureMatrix = np.array([
        [1, 1, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ])

    # Parameters:
    #  * K: number of OFDM subcarriers (actually, FFT length)
    #  * CP: length of the cyclic prefix.
    #  * pilotCarriers: array with the positions of the pilot carriers.
    #       Notice carriers are numbered from 0 (DC) to K - 1.
    #  * pilotSymbols: symbols to be placed at each pilot carrier. There
    #       should be exactly the same number of pilot symbols as pilot
    #       carriers. Otherwise, a ValueError exception is raised.
    #   * controlCarriers: list of carriers used for control information.
    #       Notice this list can be empty.
    #   * numberOfGuardCarriers: number of guardCarriers to be added to the extremes
    #       of the carrier list. The number of guard carriers specified here
    #       is used both at the beginning and at the end of the carrier list.
    #   * dataModulation: string specifying the modulation used for data carriers.
    #       Should be one of the following: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM.
    #   * controlModulation: string specifying the modulation used for control carriers.
    #       Should be one of the following: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM.
    #   * Channel response: the impulse response of the wireless channel.
    #   * SNR: SNR at the receiver in dB.
    #   * data: array of ASCII characters to be sent as the payload of the transmitted packet.
    #       If empty, data will be generated randomly for each packet.
    #   * payloadLength: Length of the payload of the transmitted packet in bits. This does not
    #       include the the control bits or the CRC32 added for integrity
    #       checking. If 'data' is empty, this many bits will be generated randomly for each
    #       new packet. Otherwise, this many bits will be taken from 'data' to generate the packet.
    #       If 'data' is not large enough, its bits will be repeated until filling the specified
    #       size.
    #   * control: bits (as an array of integers 0 or 1) that compose the information sent as
    #       the control portion of the packet.
    #   * collidingPayloadLength: payload length in bits of the colliding packet. A value of zero means no
    #       colliding packet will be simulated. All other configurations (e.g., data modulation)
    #       are the same as those of the main transmitter. However, payload and control
    #       are *always* generated randomly for the colliding transmitter. Moreover, a random offset is
    #       applied to the start of the colliding transmission. The maximum possible offset is chosen such
    #       that both transmitters end at the same time.
    #   * collidingGain: power difference between the main transmitter's signal and the colliding
    #       signal in dB. Positive values denote a higher power for the colliding signal.
    #   * dataCoding: string specifying the coding ratio for data bits. It can be "1/1", "1/2", "2/3" or "3/4".
    #       "1/1" means no coding. All others are convolutional codes based on IEEE 802.11a.
    #   * controlCoding: string specifying the coding ratio for control bits. It can be "1/1", "1/2", "2/3" or "3/4".
    #       "1/1" means no coding. All others are convolutional codes based on IEEE 802.11a.
    def __init__(self, K = 64, CP = 16, 
                    ##802.11g
                    pilotCarriers = np.array([7, 21, 44, 58]),
                    pilotSymbols = [-3-3j, -3+3j, 3+3j, 3-3j],
                    ##802.11n e 802.11ac - 20MHz
                    #pilotCarriers = np.array([7, 21, 44, 58]),
                    #pilotSymbols = [-3-3j, -3+3j, 3+3j, 3-3j],
                    ##802.11n e 802.11ac - 40MHz
                    #pilotCarriers = np.array([11, 25, 53, 76, 104, 118]),
                    #pilotSymbols = [-3-3j, -3+3j, 3+3j, 3-3j, -3-3j, -3+3j],
                    ##802.11c - 80MHz
                    #pilotCarriers = np.array([11, 39, 75, 103, 154, 182, 218, 246]),
                    #pilotSymbols = [-3-3j, -3+3j, 3+3j, 3-3j, -3-3j, -3+3j, 3+3j, 3-3j],
                    ##802.11c - 160MHz
                    #pilotCarriers = np.array([25, 53, 89, 117, 139, 167, 203, 231, 282, 310, 346, 374, 396, 424, 460, 488]),
                    #pilotSymbols = [-3-3j, -3+3j, 3+3j, 3-3j, -3-3j, -3+3j, 3+3j, 3-3j, -3-3j, -3+3j, 3+3j, 3-3j, -3-3j, -3+3j, 3+3j, 3-3j],
                    controlCarriers = np.array([]),
                    numberOfGuardCarriers = 6, dataModulation = "QPSK", controlModulation = "BPSK",
                    #channelResponse = [1, 0, 0.3+0.3j],
                    channelResponse = [1],
                    SNR = 6, data = ['a', 'b', 'c', 'd', 'e'], payloadLength = 1920,
                    control = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
                    collidingPayloadLength = 0, collidingGain = 0, dataCoding = "1/1", controlCoding = "1/2"):

        self.K = K
        self.CP = CP

        if len(pilotCarriers) <> len(pilotSymbols):
            raise ValueError("Length of pilotCarriers and pilotSymbols must match!")

        self.allCarriers = np.arange(0, self.K + 1)
        guardCarriers = np.hstack((np.arange(0, numberOfGuardCarriers + 1), np.arange(self.K, self.K - numberOfGuardCarriers, -1)))

        self.pilotCarriers = pilotCarriers
        self.pilotSymbols = pilotSymbols

        self.controlCarriers = controlCarriers

        # data carriers are all remaining carriers
        self.dataCarriers = np.setdiff1d(np.setdiff1d(np.setdiff1d(self.allCarriers, self.pilotCarriers), self.controlCarriers), guardCarriers)

        # Modulation mapping tables
        self.dataMappingTable = self.stringModulationToMappingTable(dataModulation)
        self.controlMappingTable = self.stringModulationToMappingTable(controlModulation)

        # Check if all specified modulations are valid.
        if self.dataMappingTable == None or self.controlMappingTable == None:
            raise ValueError("Unsupported modulation scheme!")

        # Compute the number of bits per symbol of each modulation
        self.dataMu = int(np.log2(len(self.dataMappingTable)))
        self.controlMu = int(np.log2(len(self.controlMappingTable)))

        # Demodulation mapping tables
        self.dataDemappingTable = {v : k for k, v in self.dataMappingTable.items()}
        self.controlDemappingTable = {v : k for k, v in self.controlMappingTable.items()}

        # Channel response
        self.channelResponse = channelResponse

        # Channel SNR
        self.SNR = SNR

        # Data carried in the packet (if any)
        self.data = data
        self.payloadLength = payloadLength

        ##
        # Coding

        # The same trellis is used for all coding rates and for both data and control
        memory = np.array([6])
        g_matrix = np.array([[0o133, 0o171]])
        self.codingTrellis = cc.Trellis(memory, g_matrix)

        # We will also pre-compute a codeword dictionary for the viterbi decoder.
        self.dictionary = self.preComputeCodewordDictionary(6, [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1]])

        # Data
        if dataCoding == "1/1":
            self.dataCodingPuncturingMatrix = None
        elif dataCoding == "1/2":
            self.dataCodingPuncturingMatrix = self.coding12PunctureMatrix
        elif dataCoding == "2/3":
            self.dataCodingPuncturingMatrix = self.coding23PunctureMatrix
        elif dataCoding == "3/4":
            self.dataCodingPuncturingMatrix = self.coding34PunctureMatrix
        elif dataCoding == "5/6":
            self.dataCodingPuncturingMatrix = self.coding56PunctureMatrix
        elif dataCoding == "7/8":
            self.dataCodingPuncturingMatrix = self.coding78PunctureMatrix
        else:
            raise ValueError("Invalid coding for data!")

        # Control
        if controlCoding == "1/1":
            self.controlCodingPuncturingMatrix = None
        elif controlCoding == "1/2":
            self.controlCodingPuncturingMatrix = self.coding12PunctureMatrix
        elif controlCoding == "2/3":
            self.controlCodingPuncturingMatrix = self.coding23PunctureMatrix
        elif controlCoding == "3/4":
            self.controlCodingPuncturingMatrix = self.coding34PunctureMatrix
        elif controlCoding == "5/6":
            self.controlCodingPuncturingMatrix = self.coding56PunctureMatrix
        elif controlCoding == "7/8":
            self.controlCodingPuncturingMatrix = self.coding78PunctureMatrix
        else:
            raise ValueError("Invalid coding for control!")

        ##
        # Control information sent along the data (if any). We can pre-compute it to avoid doing
        # it at every simulation. That means padding it to fit the number of OFDM symbols required for
        # data, as well as computing coding and a CRC for integrity checking.

        # We start calculating the number of OFDM symbols required for the data.
        numberOfOFDMSymbolsForData = int(np.ceil(self.computeLengthWithCoding(self.payloadLength + 32, self.dataCodingPuncturingMatrix) * 1.0 / (len(self.dataCarriers) * self.dataMu)))

        # Now we do the same for the control data. Notice here that 'control' is already in bits and that
        # we will perhaps use some coding and add a CRC4 for it.
        numberOfOFDMSymbolsForControl = int(np.ceil((self.computeLengthWithCoding(len(control) + 4.0, self.controlCodingPuncturingMatrix))/ (len(self.controlCarriers) * self.controlMu)))

        # Is the number of required OFDM symbols for data enough to carry the control bits?
        if numberOfOFDMSymbolsForData < numberOfOFDMSymbolsForControl:
            raise ValueError("Data is too small for the amount of control info, given modulations and carrier allocation!")

        # Compute the required padding (if any).
        targetControlLengthWithCoding = numberOfOFDMSymbolsForData * len(self.controlCarriers) * self.controlMu
        targetControlLengthWithoutCoding = self.computeLengthWithoutCoding(targetControlLengthWithCoding, self.controlCodingPuncturingMatrix)

        neededPaddingBits = targetControlLengthWithoutCoding - (len(control) + 4)
        neededPaddingReps = neededPaddingBits // len(control)
        neededPaddingRemainer = neededPaddingBits - neededPaddingReps * len(control)
        self.control = control + control*neededPaddingReps + control[0:neededPaddingRemainer]

        # Compute and append a CRC4
        self.control = np.hstack((self.control, self.crc(self.control, self.crc4Poly)))

        # Code control bits, if needed
        self.convolutionalDecoderInit()
        self.uncodedControl = self.control
        if self.controlCodingPuncturingMatrix is not None:
            self.control = self.convolutionalCode(self.control, self.controlCodingPuncturingMatrix)

        ##
        # Colliding signal
        if collidingPayloadLength > self.payloadLength:
            raise ValueError("Colliding packet cannot be larger than main transmitter's packet!")
        self.collidingPayloadLength = collidingPayloadLength
        self.collidingGain = collidingGain

    def preComputeCodewordDictionary(self, k, listOfPolys):

        numberOfCodewords = 2**(k + 1)
        dictionary = []
        currentInput = np.array([0]*(k + 1))

        for i in range(0, numberOfCodewords):
            outputBits = []
            for p in listOfPolys:
                nextOutputBit = (p * currentInput).sum() % 2
                outputBits.append(nextOutputBit)

            dictionary.append(np.array(outputBits))

            for j in range(k, -1, -1):
                currentInput[j] = (currentInput[j] + 1) % 2
                if currentInput[j] == 1:
                    break

        return dictionary

    def viterbiCalculateDistance(self, r_codeword, e_codeword):

        a = r_codeword[0] - e_codeword[0]
        if a < 0:
            a = -a
        b = r_codeword[1] - e_codeword[1]
        if b < 0:
            b = -b
        return a+b

    def viterbiDecode(self, codedBits, l):

        # Fixed parameters: k = 6; r = 2
        k = 6
        r = 2

        # Useful constants.
        inf = len(codedBits) + 1
        numberOfStates = 2**k
        inputUnit = numberOfStates

        # Initialization
        pathMetric = np.array([inf] * numberOfStates)
        pathMetric[0] = 0
        nextPathMetric = np.array([inf] * numberOfStates)
        bestPaths = [0]*numberOfStates
        nextBestPaths = [0]*numberOfStates
        numberOfDecodedBits = 0

        # Main loop.
        for i in range(0, len(codedBits), 2):
            # Get the next set of (r) received bits
            r_codeword = codedBits[i:(i+r)]

            # Iterate through all states at the previous step to generate
            # the path metrics for the current step.
            for s in range(0, numberOfStates):

                if pathMetric[s] >= inf:
                    continue

                # Each state can transition to two new states, depending if
                # the original input was 0 or 1.
                # We start with 0.
                targetState = (s >> 1)

                # Compute the expected codeword for that transition.
                e_codeword = self.dictionary[s]

                # Compute the metric for that transition
                transitionMetric = self.viterbiCalculateDistance(r_codeword, e_codeword)

                # Is this path an improvement over the previously known?
                if pathMetric[s] + transitionMetric < nextPathMetric[targetState]:
                    # Yes, update.
                    nextPathMetric[targetState] = pathMetric[s] + transitionMetric
                    nextBestPaths[targetState] = (bestPaths[s] << 1)

                # Repeat it for 1.
                targetState = targetState + (inputUnit >> 1)

                # Compute the expected codeword for that transition.
                e_codeword = self.dictionary[s + inputUnit]

                # Compute the metric for that transition
                transitionMetric = self.viterbiCalculateDistance(r_codeword, e_codeword)

                # Is this path an improvement over the previously known?
                if pathMetric[s] + transitionMetric < nextPathMetric[targetState]:
                    # Yes, update.
                    nextPathMetric[targetState] = pathMetric[s] + transitionMetric
                    nextBestPaths[targetState] = (bestPaths[s] << 1) + 1

            # Exchange current and next variables
            pathMetric = nextPathMetric
            bestPaths = nextBestPaths

            # Reinitialize next variables
            nextBestPaths = [0]*numberOfStates
            nextPathMetric = np.array([inf] * numberOfStates)

            numberOfDecodedBits = numberOfDecodedBits + 1

        # End of the main loop. Find the state with the minimum cost.
        minimumCost = inf
        bestPath = None
        for s in range(0, numberOfStates):
            if pathMetric[s] < minimumCost:
                minimumCost = pathMetric[s]
                bestPath = bestPaths[s]

        # Remove the last few decoded bits that are not part of the original message
        bestPath = bestPath >> (numberOfDecodedBits - l)

        # Translate the rest to an array form.
        output = np.array([0]*l)
        for i in range(l-1, -1, -1):
            output[i] = bestPath % 2
            bestPath = bestPath >> 1

        #print output
        return output

    def computeLengthWithCoding(self, originalLength, punctureMatrix):
        # We start by checking whether there is a puncturing matrix. If not,
        # our work is finished.
        if punctureMatrix is None:
            return originalLength

        # Coding itself simply doubles the number of bits and adds 12 more
        # after that (flush bits).
        lengthWithCoding = originalLength * 2 + 12

        # Now we have to subtract the bits removed by puncturing.
        # Let's generate an alternative, linear version of the puncturing matrix.
        punctureArray = np.reshape(np.array(punctureMatrix), -1, order='F')

        # There is a puncturing matrix. We need to check how many bits it
        # comprises, as well as how many it removes.
        totalBitsInMatrix = len(punctureArray)
        removedBitsInMatrix = totalBitsInMatrix - punctureArray.sum()

        # Now we check how many complete groups of 'totalBitsInMatrix' there
        # are in lengthWithCoding.
        completeGroups = lengthWithCoding // totalBitsInMatrix

        # The remaining bits, which do not form a complete group, have to be treated
        # differently. We need to check how many are there and then see how many of
        # them will be removed.
        lengthOfLastGroup = int(lengthWithCoding - completeGroups * totalBitsInMatrix)
        removedBitsInLastGroup = lengthOfLastGroup - punctureArray[0:lengthOfLastGroup].sum()

        # Now we compute the overall number of coded bits.
        return lengthWithCoding - removedBitsInMatrix * completeGroups - removedBitsInLastGroup

    def computeLengthWithoutCoding(self, targetNumberOfBits, punctureMatrix):
        # We start by checking whether there is a puncturing matrix. If not,
        # our work is finished.
        if punctureMatrix is None:
            return targetNumberOfBits

        # We first compute a rough approximation, based on the base code rate (1/2)
        # and the rate of bits removed by puncturing.
        # Let's generate an alternative, linear version of the puncturing matrix.
        punctureArray = np.reshape(np.array(punctureMatrix), -1, order='F')

        # There is a puncturing matrix. We need to check how many bits it
        # comprises, as well as how many it removes.
        totalBitsInMatrix = len(punctureArray)
        bitsLeftInMatrix = punctureArray.sum()

        # Now, compute the first approximation.
        approximation = int(np.floor((targetNumberOfBits / 2.0) / (float(bitsLeftInMatrix) / totalBitsInMatrix)))

        # Now, we need to refine this approximation. We perform a search procedure for that.
        # We start by checking how far we are (and to which direction).
        lengthWithCoding = self.computeLengthWithCoding(approximation, punctureMatrix)
        diff = targetNumberOfBits - lengthWithCoding
        if diff == 0:
            # Perfect match.
            return approximation
        elif diff > 0:
            direction = 1
        else:
            direction = -1
        while direction * lengthWithCoding < direction * targetNumberOfBits:
            approximation = approximation + direction
            lengthWithCoding = self.computeLengthWithCoding(approximation, punctureMatrix)

        if lengthWithCoding == lengthWithCoding:
            # Perfect match.
            return approximation
        else:
            return approximation - direction

    def convolutionalCode(self, data, punctureMatrix):
        codedData = cc.conv_encode(data, self.codingTrellis)
        if punctureMatrix is not None and len(punctureMatrix) > 1:
            cols = len(punctureMatrix[0])
            output = []
            for i in range(0, len(codedData) // 2):
                if punctureMatrix[0, i % cols] == 1:
                    output.append(codedData[2 * i])
                if punctureMatrix[1, i % cols] == 1:
                    output.append(codedData[2 * i + 1])
            return np.array(output)
        else:
            return codedData

    def convolutionalDecoderInit(self):

        self.feclib = ctypes.CDLL("./libfec.so")
        self.feclib.create_viterbi27.argtypes = (ctypes.c_int,)
        self.feclib.create_viterbi27.restype = ctypes.c_voidp
        self.feclib.init_viterbi27.argtypes = (ctypes.c_voidp, ctypes.c_int)
        self.feclib.update_viterbi27_blk.argtypes = (ctypes.c_voidp, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int)
        self.feclib.chainback_viterbi27.argtypes = (ctypes.c_voidp, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int)

    def convolutionalDecode(self, codedData, punctureMatrix, l):
        if punctureMatrix is not None and len(punctureMatrix) > 1:
            codedLengthNoPuncturing = 2*l + 12
            cols = len(punctureMatrix[0])
            input = [127]*codedLengthNoPuncturing
            inputIndex = 0
            codedDataIndex = 0
            while codedDataIndex < len(codedData):
                if punctureMatrix[inputIndex % 2, (inputIndex // 2) % cols] == 1:
                    input[inputIndex] = 255*codedData[codedDataIndex]
                    codedDataIndex = codedDataIndex + 1
                inputIndex = inputIndex + 1
            input = np.array(input)
        else:
            input = 255*codedData

        #print input
        vp = self.feclib.create_viterbi27(l)

        symbolsArrayType = ctypes.c_uint8 * len(input)
        symbolsArray = symbolsArrayType(*input)
        self.feclib.update_viterbi27_blk(vp, symbolsArray, ctypes.c_int(l + 6))

        dataArrayType = ctypes.c_uint8 * l
        dataArray = dataArrayType(*([0]*l))
        self.feclib.chainback_viterbi27(vp, dataArray, ctypes.c_int(l), ctypes.c_int(0))
        decodedData = np.array([0]*l)
        byteIndex = 0
        bitIndex = 7
        for i in range(0, l):
            #print dataArray[byteIndex]
            if dataArray[byteIndex]  & (1 << bitIndex) == 0:
                decodedData[i] = 0
            else:
                decodedData[i] = 1
            bitIndex = bitIndex - 1
            if bitIndex < 0:
                bitIndex = 7
                byteIndex = byteIndex + 1

        #print decodedData
        #sys.exit(0)
        del vp
        del dataArray
        del symbolsArray

        return decodedData

    # Very simplified implementation of a CRC. 'data' and 'poly' are both represented as arrays
    # of integers, with each position either 0 or 1.
    # TODO: improve this.
    def crc(self, data, poly):
        d = len(poly) - 1
        augmentedData = np.array(data + [0]*d)

        for i in range(0, len(data)):

            if augmentedData[i] == 1:
                augmentedData[i:(i + d + 1)] = np.add(augmentedData[i:(i + d + 1)], poly) % 2

        return augmentedData[-d:]

    def stringModulationToMappingTable(self, stringModulation):

        if stringModulation == "BPSK":
            return self.mappingTableBPSK
        elif stringModulation == "QPSK":
            return self.mappingTableQPSK
        elif stringModulation == "8PSK":
            return self.mappingTable8PSK
        elif stringModulation == "16QAM":
            return self.mappingTable16QAM
        elif stringModulation == "64QAM":
            return self.mappingTable64QAM
        elif stringModulation == "256QAM":
            return self.mappingTable256QAM
        else:
            return None

    # Generates a random payload of 'l' bits
    def generateRandomPayload(self, l):
        return np.array(np.random.binomial(n=1, p=0.5, size=(l)))

    # Generates a payload with l bits from self.data
    def generatePayloadFromData(self, l):
        dataBits = []
        for x in self.data:
            binary = bin(ord(x))
            for i in range(2, 9):
                dataBits.append(int(binary[i]))

        reps = l // len(dataBits)
        remaining = l % len(dataBits)

        return np.array(dataBits * reps + dataBits[0:remaining])

    # Groups bits into carrier symbols, considering the number of bits per
    # symbol used in the modulation.
    def groupBitsIntoSymbols(self, bits, mu):
        if len(bits) % mu > 0:
            paddedBits = np.hstack((bits, [0]*(mu - (len(bits) % mu))))
            return paddedBits.reshape((-1, mu))
        else:
            return bits.reshape((-1, mu))

    # Map each bit group of the input to the corresponding symbol, according
    # to the mapping table of the modulation.
    def mapBitsToSymbols(self, bitGroups, mappingTable):
        return np.array([mappingTable[tuple(g)] for g in bitGroups])

    # Generate an OFDM symbol based on the data and control symbols provided
    # as input.
    def generateOFDMSymbol(self, dataSymbols, controlSymbols):
        # initialize the overall K subcarriers
        symbol = np.zeros(self.K, dtype=complex)
        # Fill the pilot carriers
        symbol[self.pilotCarriers] = self.pilotSymbols
        # Fill the control carriers (if it exists)
        if len(self.controlCarriers) <> 0:
            symbol[self.controlCarriers] = controlSymbols
        # Fill the data carriers
        symbol[self.dataCarriers] = dataSymbols
        return symbol

    # Compute the IDFT of the list of OFDM symbols, resulting in time samples.
    def IDFT(self, OFDMSymbols):
        return np.fft.ifft(OFDMSymbols)

    # Add a cyclic prefix to the time samples of each OFDM symbol in the input.
    def addCP(self, OFDMTimeSamples):
        output = []
        for i in range(0, len(OFDMTimeSamples)):
            cp = OFDMTimeSamples[i][-self.CP:]
            if output == []:
                output = np.hstack((cp, OFDMTimeSamples[i]))
            else:
                output = np.vstack((output, np.hstack((cp, OFDMTimeSamples[i]))))
        return(output)

    # Apply a channel model with a randomly generated AWGN and the configured channel response.
    def channel(self, signal):
        convolved = np.convolve(signal, self.channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-self.SNR/10.0)  # calculate noise power based on signal power and SNR

        #print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

        # Generate complex noise with given variance
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return (convolved + noise)[0:len(signal)]

    # Remove the cyclic prefix of each OFDM sample at the receiver
    def removeCP(self, OFDMTimeSamples):
        output = []
        for i in range(0, len(OFDMTimeSamples)):
            if output == []:
                output = OFDMTimeSamples[i][self.CP:(self.CP + self.K)]
            else:
                output = np.vstack((output, OFDMTimeSamples[i][self.CP:(self.CP + self.K)]))
        return(output)

    # Apply a DFT to each group of time samples to generate OFDM symbols in the frequency domain.
    def DFT(self, OFDMTimeSamples):
        return np.fft.fft(OFDMTimeSamples)

    # Estimate channel parameters based on the received OFDM symbol for equalization
    def channelEstimate(self, OFDMSymbol):
        pilots = OFDMSymbol[self.pilotCarriers]  # extract the pilot values from the RX signal
        Hest_at_pilots = pilots / self.pilotSymbols # divide by the transmitted pilot values

        # Perform interpolation between the pilot carriers to get an estimate
        # of the channel in the data carriers. Here, we interpolate absolute value and phase
        # separately
        Hest_abs = scipy.interpolate.interp1d(self.pilotCarriers, abs(Hest_at_pilots), kind='linear', fill_value="extrapolate")(self.allCarriers)
        Hest_phase = scipy.interpolate.interp1d(self.pilotCarriers, np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(self.allCarriers)
        Hest = Hest_abs * np.exp(1j*Hest_phase)

        return Hest[1:] # Ignore the DC component
        
    # diff between TX and RX pilot values and sum for EVM
    def diffPilotsTXRX(self, OFDMSymbol):
        pilots = OFDMSymbol[self.pilotCarriers]  # extract the pilot values from the RX signal
        
        diff = self.pilotSymbols-pilots
        sumDiffSymbols = 0.0
        for symbolDiff in diff:
            sumDiffSymbols = sumDiffSymbols + pow((abs(symbolDiff)),2)
        
        return sumDiffSymbols
        
    # EVM from pilot values
    def pilotsEVM(self, sumPilotsEVM, numSymbols):
        numCarriers = len(self.pilotCarriers)
        totalEVMSymbols = numSymbols * len(self.pilotCarriers)
        
        # transmission power
        P0 = 1
        EVM = math.sqrt(sumPilotsEVM / (numCarriers * totalEVMSymbols * P0))
                
        # EVM
        return EVM

    # Apply equalization based on estimated channel parameters
    def equalize(self, OFDMSymbol, Hest):
        return OFDMSymbol / Hest

    # Demultiplex data and control bits
    def demultiplexOFDMSymbol(self, OFDMSymbol):
        if len(self.controlCarriers) <> 0:
            return (OFDMSymbol[self.dataCarriers], OFDMSymbol[self.controlCarriers])
        else:
            return (OFDMSymbol[self.dataCarriers], np.array([]))

    # Demodulate a list of symbols according to the mapping table provided in the input
    def demap(self, carrierSymbols, demappingTable):
        # array of possible constellation points
        constellation = np.array([x for x in demappingTable.keys()])

        # calculate distance of each RX point to each possible point
        dists = abs(carrierSymbols.reshape((-1,1)) - constellation.reshape((1,-1)))
                
        # for each carrier symbol, choose the index in constellation
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)
        
        # get back the real constellation point
        hardDecision = constellation[const_index]
                
        # transform the constellation point into the bit groups
        return np.array([demappingTable[C] for C in hardDecision])
        
    # EVM from data and control https://ieeexplore.ieee.org/document/6310924
    def EVM(self, carrierSymbols, demappingTable, totalEVMSymbols):
        # array of possible constellation points
        constellation = np.array([x for x in demappingTable.keys()])

        # calculate distance of each RX point to each possible point
        dists = abs(carrierSymbols.reshape((-1,1)) - constellation.reshape((1,-1)))
                
        # for each carrier symbol, choose the index in constellation
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)
        
        # https://ieeexplore.ieee.org/document/6310924
        sumSymbols = 0.0
        numCarriers = len(dists)
        for x in range(len(dists)):
            sumSymbols = sumSymbols + pow((abs(dists[x][const_index[x]])),2)
        
        # transmission power
        P0 = 1
        #print(totalEVMSymbols)
        EVM = math.sqrt(sumSymbols / (numCarriers * totalEVMSymbols * P0))
        #print("%.15f" % EVM)
                
        # EVM
        return EVM

    # Wrapper function for computing the CRC32 of some data (represented as an integer array of 0s and 1s)
    def crc32(self, data):

        dataBitsAsString = "".join([str(i) for i in data])
        dataBitsAsBytes = bytes([int(dataBitsAsString[i:i+8], 2) for i in range(0, len(dataBitsAsString), 8)])
        crcAsString = format(binascii.crc32(dataBitsAsBytes) & 0xffffffff, '032b')

        return [int(crcAsString[i]) for i in range(0, 32)]

    # Generates a new packet and the corresponding TX signal. If 'colliding' = True,
    # generates a signal of a colliding node, instead of the main transmitter.
    def generateTXSignal(self, colliding=False):

        ##
        # Data
        #

        if colliding == True:
            payloadLength = self.collidingPayloadLength
            dataBits = self.generateRandomPayload(payloadLength)
            uncodedDataBits = None
        else:
            payloadLength = self.payloadLength
            # Generate the data bits.
            if len(self.data) == 0:
                dataBits = self.generateRandomPayload(payloadLength)
            else:
                dataBits = self.generatePayloadFromData(payloadLength)

            # Compute and append a CRC32 for integrity checking.
            dataBits = np.hstack((dataBits, self.crc32(dataBits)))

            # If necessary, code the data.
            uncodedDataBits = dataBits
            if self.dataCodingPuncturingMatrix is not None:
                dataBits = self.convolutionalCode(dataBits, self.dataCodingPuncturingMatrix)

        # Group bits according to the number of bits in each symbol of
        # the modulation used for data
        groupedDataBits = self.groupBitsIntoSymbols(dataBits, self.dataMu)

        # Translate data bits into a sequence of data symbols
        dataCarrierSymbols = self.mapBitsToSymbols(groupedDataBits, self.dataMappingTable)

        # Pad data symbols to match the number of data carriers if necessary
        numberOfPaddingSymbols = len(dataCarrierSymbols) % len(self.dataCarriers)
        if numberOfPaddingSymbols > 0:
            dataCarrierSymbols = np.hstack((dataCarrierSymbols, [0]*(len(self.dataCarriers) - numberOfPaddingSymbols)))

        # TODO: generate header, multiplex it along with data?

        ##
        # Control

        if colliding == True:

            # Control bits are also generated randomly for the colliding signal.
            controlBits = self.generateRandomPayload(len(self.control))
        else:

            # Control bits are already computed in binary form (including CRC).
            controlBits = self.control

        # Group bits according to the number of bits in each symbol of
        # the modulation used for control
        groupedControlBits = self.groupBitsIntoSymbols(controlBits, self.controlMu)

        # Translate control bits into a sequence of control symbols
        controlCarrierSymbols = self.mapBitsToSymbols(groupedControlBits, self.controlMappingTable)

        # Pad control symbols to match the number of OFDM symbols used by the data carrier symbols
        paddedControlCarrierSymbols = np.array([0+0j] * (len(dataCarrierSymbols) / len(self.dataCarriers)) * self.controlMu)
        if len(controlCarrierSymbols) < len(paddedControlCarrierSymbols):
            paddedControlCarrierSymbols[0:len(controlCarrierSymbols)] = controlCarrierSymbols
        else:
            paddedControlCarrierSymbols = controlCarrierSymbols[0:len(paddedControlCarrierSymbols)]
        controlCarrierSymbols = paddedControlCarrierSymbols

        # Iterate through the data symbols, generating the sequence of OFDM symbols
        OFDMSymbols = []
        dataPointer = 0
        controlPointer = 0
        while dataPointer < len(dataCarrierSymbols):
            nextOFDMSymbol = self.generateOFDMSymbol(dataCarrierSymbols[dataPointer:(dataPointer + len(self.dataCarriers))], [controlCarrierSymbols[controlPointer:(controlPointer + len(self.controlCarriers))]])
            OFDMSymbols.append(nextOFDMSymbol)
            dataPointer = dataPointer + len(self.dataCarriers)
            controlPointer = controlPointer + len(self.controlCarriers)

        # Compute the time samples associated with each OFDM symbol (sans cyclic prefix)
        OFDMTimeSamples = self.IDFT(OFDMSymbols)

        # Append the cyclic prefix to each OFDM time sample
        OFDMTimeSamples = self.addCP(OFDMTimeSamples)

        # Linearize the OFDM time samples
        txSignal = OFDMTimeSamples.reshape(-1)

        # If we are generating the colliding signal, apply the proper gain.
        if colliding == True:
            txSignal = txSignal * (10.0**(self.collidingGain / 10.0))

        return (txSignal, dataBits, uncodedDataBits)

    # Attempts to interpret a RX signal into the data and control bits. 'l' is the size of the data payload.
    def interpretRXSignal(self, rxSignal, l):

        # Divide the received signal into sequences of time samples (for each OFDM symbol with CP) again.
        OFDMTimeSamples = rxSignal.reshape((-1, self.K + self.CP))

        # Remove cyclic prefixes
        OFDMTimeSamples = self.removeCP(OFDMTimeSamples)

        # Compute the list of OFDM symbols in the frequency domain.
        OFDMSymbols = self.DFT(OFDMTimeSamples)

        dataCarrierSymbols = np.array([])
        controlCarrierSymbols = np.array([])
        sumPilotsEVM = 0
        for i in range(0, len(OFDMSymbols)):
            # Estimate channel parameters for equalization
            Hest = self.channelEstimate(OFDMSymbols[i])
            diffPilots = self.diffPilotsTXRX(OFDMSymbols[i])
            sumPilotsEVM = sumPilotsEVM + diffPilots

            # Apply equalization
            equalizedSymbol = self.equalize(OFDMSymbols[i], Hest)

            # TODO: separate header symbol, demultiplex it and parse it.

            # Demultiplex data and control bits from the equalized OFDM symbols
            (nextDataCarrierSymbols, nextControlCarrierSymbols) = self.demultiplexOFDMSymbol(equalizedSymbol)

            # Concatenate these symbols to the complete list of (data and control) symbols
            dataCarrierSymbols = np.hstack((dataCarrierSymbols, nextDataCarrierSymbols))
            controlCarrierSymbols = np.hstack((controlCarrierSymbols, nextControlCarrierSymbols))


        # Demodulate data and control bits
        receivedDataBits = self.demap(dataCarrierSymbols, self.dataDemappingTable)
        receivedControlBits = self.demap(controlCarrierSymbols, self.controlDemappingTable)
        
        # EVM
        dataEVM = self.EVM(dataCarrierSymbols, self.dataDemappingTable, len(self.dataCarriers))
        controlEVM = self.EVM(controlCarrierSymbols, self.controlDemappingTable, len(self.controlCarriers))
        pilotEVM = self.pilotsEVM(sumPilotsEVM, len(OFDMSymbols))

        # Linearize data and control bits
        receivedDataBits = receivedDataBits.reshape(-1)
        receivedControlBits = receivedControlBits.reshape(-1)

        # Remove padding.
        receivedDataBits = receivedDataBits[0:l]

        return (receivedDataBits, receivedControlBits, dataEVM, controlEVM, pilotEVM)

    # Perform the simulation of transmitting a single packet. Outputs
    # statistics regarding the transmission attempt.
    def simulate(self):

        # Generate the tx signal of the packet to be transmitted.
        (txSignal, dataBits, uncodedDataBits) = self.generateTXSignal()

        # Is there a colliding signal in this simulation? If so, generate it.
        if self.collidingPayloadLength > 0:

            collidingSignal = self.generateTXSignal(colliding=True)[0]

            if len(collidingSignal) < len(txSignal):

                maxOffset = len(txSignal) - len(collidingSignal)
                offset = randrange(maxOffset)
                padding = maxOffset - offset
                collidingSignal = np.hstack((np.array([0]*offset), collidingSignal, np.array([0]*padding)))

            txSignal = txSignal + collidingSignal

        # Introduce channel modifications (noise, distortions, etc) to the signal
        rxSignal = self.channel(txSignal)

        # Receive the packet (data and control)
        (receivedDataBits, receivedControlBits, dataEVM, controlEVM, pilotEVM) = self.interpretRXSignal(rxSignal, len(dataBits))

        ##
        # Compute statistics/results
        results = {}

        # Check BER on data
        DataBER = np.sum(abs(receivedDataBits-dataBits))/float(len(dataBits))
        results["dataBER"] = DataBER

        # Check BER on decoded data
        if self.dataCodingPuncturingMatrix is not None:
            decodedReceivedDataBits = self.convolutionalDecode(receivedDataBits, self.dataCodingPuncturingMatrix, len(uncodedDataBits))
            DataBER = np.sum(abs(decodedReceivedDataBits-uncodedDataBits))/float(len(dataBits))
        results["decodedDataBER"] = DataBER
        results["dataCRCFailure"] = 0

        if DataBER > 0:

            results["dataLoss"] = 1
            # Check if CRC matches.
            receivedDataCRC = receivedDataBits[-32:]
            computedCRC = self.crc32(receivedDataBits[0:-32])

            if np.sum(abs(receivedDataCRC-computedCRC)) == 0:
                results["dataCRCFailure"] = 1
        else:

            results["dataLoss"] = 0

        # Control might have been padded by the physical layer. Remove the padding.
        receivedControlBits = receivedControlBits[0:len(self.control)]

        # Check BER on coded control.
        ControlBER = np.sum(abs(receivedControlBits-self.control))/float(len(self.control))
        results["controlBER"] = ControlBER

        # Now decode it and check the BER again.
        if self.controlCodingPuncturingMatrix is not None:
            decodedReceivedControBits = self.convolutionalDecode(receivedControlBits, self.controlCodingPuncturingMatrix, len(self.uncodedControl))
            ControlBER = np.sum(abs(decodedReceivedControBits-self.uncodedControl))/float(len(self.uncodedControl))
        results["decodedControlBER"] = ControlBER
        results["controlCRCFailure"] = 0

        if ControlBER > 0:

            results["controlLoss"] = 1
            # Check if CRC matches.
            receivedControlCRC = receivedControlBits[-4:]
            computedCRC = self.crc(receivedControlBits[0:-4].tolist(), self.crc4Poly)

            if np.sum(abs(receivedControlCRC-computedCRC)) == 0:
                #print "CRC4 failed to catch corruption on control!"
                results["controlCRCFailure"] = 1
        else:

            results["controlLoss"] = 0
            
        results["dataEVM"] = dataEVM
        results["controlEVM"] = controlEVM
        results["pilotEVM"] = pilotEVM

        return results

def usage():

    pass

##
# Main program

# Argument parsing

parser = argparse.ArgumentParser(description='Simulate OFDM channel with collision detection')
parser.add_argument('--K', type=int, help='FFT length', default=64)
parser.add_argument('--CP', type=int, help='Length of the cyclic prefix', default=16)
parser.add_argument('--pilotCarriers', type=int, help='Array with the positions of the pilot carriers, numbered from 0 (DC) to K - 1.', nargs='+')
parser.add_argument('--pilotSymbols', type=int, help='Array of the symbols to be transmitted in the pilot carriers', nargs='+')
parser.add_argument('--controlCarriers', type=int, help='List of carriers used for control information. Can be empty', default=[32], nargs='+')
parser.add_argument('--snr', type=float, help='Link SNR in dB', default=6)
parser.add_argument('--numberOfGuardCarriers', type=int, help='Number of guardCarriers to be added to the extremes of the carrier list', default=6)
parser.add_argument('--dataModulation', help='Modulation used for data carriers. Either BPSK, QPSK, 8PSK, 16QAM, 64QAM or 256QAM.', default='QPSK')
parser.add_argument('--controlModulation', help='Modulation used for control carriers. Either BPSK, QPSK, 8PSK, 16QAM, 64QAM or 256QAM.', default='BPSK')
parser.add_argument('--channelResponse', type=float, help='Array of the impulse response of the wireless channel', nargs='+')
parser.add_argument('--payloadLength', type=int, help='Data length (without coding or CRC)', default=4100)
parser.add_argument('--collidingPayloadLength',  type=int, help='Length of the data of the colliding packet (0, if none)', default=0)
parser.add_argument('--collidingGain', type=float, help='power difference between the main transmitter\'s signal and the colliding signal in dB', default=0)
parser.add_argument('--dataCoding', help='"1/1", "1/2", "2/3" or "3/4"', default='3/4')
parser.add_argument('--controlCoding', help='"1/1", "1/2", "2/3" or "3/4"', default='1/2')
parser.add_argument('--reps', type=int, help='Number of repetitions', default=1)
opts = parser.parse_args(sys.argv[1:])

#accumulatedResults = {
#    "dataBER": 0,
#    "controlBER": 0,
#    "dataCRCFailure": 0,
#    "controlCRCFailure": 0,
#    "dataLoss": 0,
#    "controlLoss": 0,
#    "decodedControlBER": 0,
#    "decodedDataBER": 0
#}

print("=============================================")
print("var_K                      = %s" % (opts.K))
print("var_CP                     = %d" % (opts.CP))
print("var_SNR                    = %f" % (opts.snr))
print("var_numberOfGuardCarriers  = %s" % (opts.numberOfGuardCarriers))
print("var_dataModulation         = %s" % (opts.dataModulation))
print("var_controlModulation      = %s" % (opts.controlModulation))
print("var_payloadLength          = %d" % (opts.payloadLength))
print("var_collidingPayloadLength = %d" % (opts.collidingPayloadLength))
print("var_collidingGain          = %d" % (opts.collidingGain))
print("var_dataCoding             = %s" % (opts.dataCoding))
print("var_controlCoding          = %s" % (opts.controlCoding))
print("var_reps                   = %d" % (opts.reps))
print("=============================================")

totald = 0
#print(totald)
totalc = 0
#print(totalc)
totaldc = 0
#print(totaldc)

print("# dataBER\tdecodedDataBER\tdataLoss\tdataCRCFailure\tcontrolBER\tdecodedControlBER\tcontrolLoss\tcontrolCRCFailure\tdataEVM\tcontrolEVM\tpilotEVM")
for i in range(0, opts.reps):
    if i % 20 == 0:
        sim = OFDMSim(controlCarriers = np.array(opts.controlCarriers),
                K=opts.K,
                SNR=opts.snr,
                numberOfGuardCarriers=opts.numberOfGuardCarriers,
                payloadLength=opts.payloadLength,
                dataModulation=opts.dataModulation,
                controlModulation=opts.controlModulation,
                data=[],
                collidingGain=opts.collidingGain,
                collidingPayloadLength=opts.collidingPayloadLength,
                dataCoding=opts.dataCoding,
                controlCoding=opts.controlCoding)
        gc.collect()

    results = sim.simulate()
    print(" %.8f\t%.8f\t%d\t\t%d\t\t%.8f\t%.8f\t\t%d\t\t%d\t\t%.8f\t\t%.8f\t\t%.8f" % (results["dataBER"],
        results["decodedDataBER"], results["dataLoss"], results["dataCRCFailure"],
        results["controlBER"], results["decodedControlBER"], results["controlLoss"], results["controlCRCFailure"],
        results["dataEVM"], results["controlEVM"], results["pilotEVM"]))

    if results["dataLoss"] == 1 and results["controlLoss"] == 1:
        totaldc = totaldc + 1;

    totald = totald + results["dataLoss"]
    totalc = totalc + results["controlLoss"]

print("")
print("SNR: %d | Tx ent dados: %f | Tx ent controle: %f | Tx perda dados e cont: %f" %  (opts.snr, 100-(totald*100/(opts.reps)),
    100-(totalc*100/(opts.reps)), (totaldc*100/(opts.reps))))
print("===")
print("")
print("")
    #if i % 10 == 9:
    #    h = hpy()
    #    print h.heap()
    #for stat in results:
    #    accumulatedResults[stat] = accumulatedResults[stat] + results[stat]
    #accumulatedResults["dataBER"] = accumulatedResults["dataBER"] + results["dataBER"]
    #accumulatedResults["controlBER"] = accumulatedResults["controlBER"] + results["controlBER"]
    #accumulatedResults["dataCRCFailure"] = accumulatedResults["dataCRCFailure"] + results["dataCRCFailure"]
    #accumulatedResults["controlCRCFailure"] = accumulatedResults["controlCRCFailure"] + results["controlCRCFailure"]


#print accumulatedResults
#for stat in accumulatedResults:
#    print ("Average " + stat + ": ", accumulatedResults[stat] / reps)
#print ("Average dataBER: ", accumulatedResults["dataBER"] / reps)
#print ("Average controlBER: ", accumulatedResults["controlBER"] / reps)
#print ("Average dataCRCFailure: ", accumulatedResults["dataCRCFailure"] / reps)
#print ("Average datacontrolCRCFailureBER: ", accumulatedResults["controlCRCFailure"] / reps)
