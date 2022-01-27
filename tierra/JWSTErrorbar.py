import numpy as np
import os
from bisect import bisect

import matplotlib.pyplot as plt


def BinningDataNIRSpecPrism(WavelengthHS=None, ValuesHS=None, ErrorFlag=False):
    '''
    Function that estimates the error for an observed cross-section:

    Wavelength: array
                Floating array of

    ValuesHS: array
              Floating points of error

    ErrorFlag: bool
             Introduce scatter to the data based of the fit
    '''

    Parameters = "WavLow,WavUpp,WavC,BinnedNoise,Npix"
    Location = __file__.replace("JWSTErrorbar.py", "NIRSpecPrism.R100.txt")
    Data = np.genfromtxt(Location, skip_header=1, names=Parameters)

    Wavelength = Data["WavC"]
    Res = Wavelength[1:]/np.diff(Wavelength)

    WavelengthLowerErr = Data["WavC"]-Data["WavLow"]
    WavelengthUpperErr = Data["WavUpp"]-Data["WavC"]

    if np.mean(ValuesHS)<1:
        print("Converting the cross-section into ppm.")
        ValuesHS*=1e6

    InterpolatedCS = np.zeros(len(Wavelength))
    for counter in range(len(Wavelength)):
        StartIndex = bisect(WavelengthHS, Data['WavLow'][counter])
        StopIndex = bisect(WavelengthHS, Data['WavUpp'][counter])
        InterpolatedCS[counter] = np.mean(ValuesHS[StartIndex:StopIndex])

    if ErrorFlag:
        Error2Add = np.zeros(len(Wavelength))
        for counter in range(len(Wavelength)):
            Error2Add[counter] = np.random.normal(0,Data['BinnedNoise'][counter],1)[0]
        InterpolatedCS+=Error2Add

    return Data["WavLow"], Data["WavUpp"], Wavelength, InterpolatedCS, Data['BinnedNoise']



def BinningDataCombined(WavelengthHS=None, RValue=100, ValuesHS=None, ErrorFlag=False, PlanetType="Earth"):
    '''
    Function that estimates the error for an observed cross-section:

    Wavelength: array
                Floating array of

    ValuesHS: array
              Floating points of error

    ErrorFlag: bool
             Introduce scatter to the data based of the fit
    '''

    print("Inside the binning data Combined")

    Parameters = "WavLow,WavUpp,WavC,BinnedNoise,Npix"

    if PlanetType.upper()=="EARTH":
        print("Using super-earth value")
        Parameters = "WavLow,WavUpp,WavC,BinnedNoise,Npix"
        Location = __file__.replace("JWSTErrorbar.py", "Combined.R%s.SE.txt" %str(int(RValue)))

    elif PlanetType.upper()=="HJ":
        print("Using warm-Jupiter value")
        Parameters = "WavLow,WavUpp,WavC,BinnedNoise,Npix"
        Location = __file__.replace("JWSTErrorbar.py", "Combined.R%s.HJ.txt" %str(int(RValue)))
    else:
        print("No such binning scheme is available")
        assert 1==0

    Data = np.genfromtxt(Location, skip_header=1, names=Parameters)
    Wavelength = Data["WavC"]
    Res = Wavelength[1:]/np.diff(Wavelength)

    WavelengthLowerErr = Data["WavC"]-Data["WavLow"]
    WavelengthUpperErr = Data["WavUpp"]-Data["WavC"]

    if np.mean(ValuesHS)<1:
        print("Converting the cross-section into ppm.")
        ValuesHS*=1e6

    InterpolatedCS = np.zeros(len(Wavelength))
    for counter in range(len(Wavelength)):
        StartIndex = bisect(WavelengthHS, Data['WavLow'][counter])
        StopIndex = bisect(WavelengthHS, Data['WavUpp'][counter])
        InterpolatedCS[counter] = np.mean(ValuesHS[StartIndex:StopIndex])

    if ErrorFlag:
        Error2Add = np.zeros(len(Wavelength))
        for counter in range(len(Wavelength)):
            Error2Add[counter] = np.random.normal(0,Data['BinnedNoise'][counter],1)[0]
        InterpolatedCS+=Error2Add

    return Data["WavLow"], Data["WavUpp"], Wavelength, InterpolatedCS, Data['BinnedNoise']
