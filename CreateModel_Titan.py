import matplotlib.pyplot as plt
import numpy as np
import os
from tierra import Target
from tierra.transmission import TransmissionSpectroscopy
from tierra.JWSTErrorbar import BinningDataNIRSpecPrism, BinningDataCombined

PlanetParamsDict = {}
PlanetParamsDict['P0'] = 1.0 #the base of the atmosphere being probedatmosphere
PlanetParamsDict['T0'] = 160
PlanetParamsDict['ALR'] = 0.01
PlanetParamsDict['TInf'] = 160
PlanetParamsDict['Mass'] = 0.0225   # Need to find proper references for mass
PlanetParamsDict['Radius'] = 0.404  # Need to find proper references for mass
PlanetParamsDict['MR_CO'] = 1e-7
PlanetParamsDict['MR_CO2'] = 1e-50
PlanetParamsDict['MR_H2O'] = 1e-50
PlanetParamsDict['MR_CH4'] = 1e-5
PlanetParamsDict['MR_O3'] = 1e-50
PlanetParamsDict['MR_N2'] = 0.98


#Currently the hydrogen makes up the rest of the composition so that it adds up to 1
MR_H2 = 85./100.*(1.0 -PlanetParamsDict['MR_CO'] -PlanetParamsDict['MR_CO2']  \
                -PlanetParamsDict['MR_H2O']-PlanetParamsDict['MR_CH4']  \
                -PlanetParamsDict['MR_O3']-PlanetParamsDict['MR_N2'])

PlanetParamsDict['MR_H2'] = MR_H2


#Get the number density
TotalNumbers = 6.023e23*PlanetParamsDict['P0']*273.15/PlanetParamsDict['T0']


N0_N2Log = np.log10(PlanetParamsDict['MR_N2']*TotalNumbers)
N0_COLog = np.log10(PlanetParamsDict['MR_CO']*TotalNumbers)
N0_H2OLog = np.log10(PlanetParamsDict['MR_H2O']*TotalNumbers)
N0_CO2Log = np.log10(PlanetParamsDict['MR_CO2']*TotalNumbers)
N0_CH4Log = np.log10(PlanetParamsDict['MR_CH4']*TotalNumbers)
N0_O3Log = np.log10(PlanetParamsDict['MR_O3']*TotalNumbers)
N0_H2Log = np.log10(PlanetParamsDict['MR_H2']*TotalNumbers)


StellarParamsDict = {}
StellarParamsDict['Mass'] = 0.55
StellarParamsDict['Radius'] = 0.55
StellarParamsDict['TEff'] = 5500 #Temperature in K ---> Not really important for TITAN
StellarParamsDict['Fe/H'] = 0.0 #Metallicity of the star
StellarParamsDict['JMag']= 10.0 #JMag

Planet1 = Target.System(PlanetParamsDict=PlanetParamsDict, StellarParamsDict=StellarParamsDict, LoadFromFile=False)

Planet1.PT_Profile(zStep=0.1, ShowPlot=False)

print("The mean molecular mass of the atmosphere is:", Planet1.mu)


CurrentDirectory = os.getcwd()
print("The current directory is given by::", CurrentDirectory)
input("We wait here...")
BaseLocation =  os.path.join(CurrentDirectory,"CS")
Planet1.LoadCrossSection(BaseLocation, SubFolder="CS_1", CIA=False)
T1 = TransmissionSpectroscopy(Planet1, CIA=False)
T1.CalculateTransmission(Planet1)

XValue = Planet1.WavelengthArray*1e4
YValue = T1.SpectrumHeight

plt.figure(figsize=(12,8))
plt.plot(XValue, YValue/1e5, "k-")
plt.xlabel("Wavelength (microns)", fontsize=20)
plt.ylabel("Atmospheric Height (km)", fontsize=20)
plt.savefig("Figures/Titan.png")
plt.close()
