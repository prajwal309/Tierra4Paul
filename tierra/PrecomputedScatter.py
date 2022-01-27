import numpy as np
import os
from bisect import bisect

def BinningDataNIRSpecPrism(WavelengthHS=None, ValuesHS=None):
    input("Wait here...")
    Parameters = "WavLow,WavUpp,WavC,BinnedNoise,Npix"
    print(os.path.getcwd())
    input("Wait here please...")
    Data = np.genfromtxt("./NIRSpecPrism.R100.txt", skip_header=1, names=Parameters)

    Wavelength = Data["WavC"]
    Res = Wavelength[1:]/np.diff(Wavelength)

    InterpolatedValues = np.zeros(Wavelength)


    for counter in range(len(Wavelength)):
        print(counter)
        StartIndex = bisect(WavelengthHS, Data['WavLow'][counter])
        StopIndex = bisect(WavelengthHS, Data['WavUpp'][counter])
        InterpolatedCS[counter] = np.mean(ValuesHS[StartIndex:StopIndex])

    return Wavelength, InterpolatedValues, Data['BinnedNoise']
