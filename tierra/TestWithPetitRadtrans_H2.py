from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import numpy as np
import matplotlib.pyplot as plt

P0_Ref = 0.98 #Bar
T_Ref = 370.0

H2O_Ref = 0
CO2_Ref = 0
CO_Ref = 1e-5

MoleculeSave = ""

if H2O_Ref>0:
    MoleculeSave += "_H2O"
if CO2_Ref>0:
    MoleculeSave += "_CO2"
if CO2_Ref>0:
    MoleculeSave += "_CO"

H2_Ref = 0.912
He_Ref = 0.087

############################################
############################################
#Now calculate the cross section
import os
from tierra import Target
from tierra.transmission import TransmissionSpectroscopy
from tierra.JWSTErrorbar import BinningDataNIRSpecPrism, BinningDataCombined


def ParsePlanetFile():
    '''
    This function parses the planetary file
    '''
    PlanetParams = {}
    if os.path.exists("PlanetParam.HJ.ini"):
        FileContent = open("PlanetParam.HJ.ini", "r").readlines()
        for Line in FileContent:
            Item = Line.split("#")[0].replace(" ","")
            key, Value = Item.split(":")
            PlanetParams[key]=float(Value)
    else:
        print("PlanetParam.ini does not exist in the local dictionary")
    return PlanetParams


def ParseStarFile():
    '''
    This function parses the star file i.e StelarParam.ini
    '''
    StellarParams = {}
    if os.path.exists("StellarParam.HJ.ini"):
        FileContent = open("StellarParam.HJ.ini", "r").readlines()
        for Line in FileContent:
            Item = Line.split("#")[0].replace(" ","")
            key, Value = Item.split(":")
            StellarParams[key]=float(Value)
    else:
        print("StellarParam.HJ.ini does not exist in the local dictionary")
    return StellarParams



PlanetParamsDict = ParsePlanetFile()
StellarParamsDict = ParseStarFile()

BaseLocation =  "/media/prajwal/a66433b1-e5b2-467e-8ebf-5857f498dfce/LowerResolutionData/Trimmed_n_Rayleigh_100_000"
#BaseLocation =  "/media/prajwal/a66433b1-e5b2-467e-8ebf-5857f498dfce/LowerResolutionData/R1000"

PlanetParamsDict['P0'] = P0_Ref*0.986923 #atmosphere
PlanetParamsDict['T0'] = T_Ref
PlanetParamsDict['ALR'] = 0.00001
PlanetParamsDict['TInf'] = T_Ref
PlanetParamsDict['Mass'] = 317.8
PlanetParamsDict['Radius'] = 11.21
PlanetParamsDict['MR_CO'] = CO_Ref
PlanetParamsDict['MR_CO2'] = CO2_Ref
PlanetParamsDict['MR_H2O'] = H2O_Ref
PlanetParamsDict['MR_CH4'] = 0.000
PlanetParamsDict['MR_O3'] = 0.0
PlanetParamsDict['MR_N2'] = 0.0
PlanetParamsDict['MR_H2'] = 0.912
Planet1 = Target.System(PlanetParamsDict, StellarParamsDict, LoadFromFile=False)

Planet1.PT_Profile(zStep=0.1, ShowPlot=False)

from scipy.interpolate import interp1d


print("The mean molecular mass is:", Planet1.mu)

Planet1.LoadCrossSection(BaseLocation, SubFolder="CS_1", CIA=True)
T1 = TransmissionSpectroscopy(Planet1, CIA=True)
T1.CalculateTransmission(Planet1, interpolation='bilinear')

T1_False = TransmissionSpectroscopy(Planet1, CIA=True)
T1_False.CalculateTransmission(Planet1, interpolation='bilinear')


########################################################################
########################################################################

MATLAB_loc = "/media/prajwal/b3feb060-a565-44ab-a81b-7dd59881cba0/Simulate_T_spectrum/data"
MATLAB_z = np.loadtxt(MATLAB_loc+"/z_.txt", delimiter=",")*1e5  #Converting km to cm
MATLAB_zcm = np.loadtxt(MATLAB_loc+"/z_.txt", delimiter=",")
MATLAB_Temp = np.loadtxt(MATLAB_loc+"/T_z.txt", delimiter=",")
MATLAB_Pr = np.loadtxt(MATLAB_loc+"/P_z.txt", delimiter=",")
MATLAB_Spectrum = np.loadtxt(MATLAB_loc+"/Spectrum_.txt", delimiter=",")
MATLAB_Spectrum /= Planet1.Rs**2
MATLAB_Lam = np.loadtxt(MATLAB_loc+"/Wavelength_.txt", delimiter=",")
MATLAB_xNew = np.loadtxt(MATLAB_loc+"/xNew_.txt", delimiter=",")
MATLAB_x__ = np.loadtxt(MATLAB_loc+"/x__.txt", delimiter=",")
MATLAB_ds = np.loadtxt(MATLAB_loc+"/ds_.txt", delimiter=",")
MATLAB_nz = np.loadtxt(MATLAB_loc+"/n_z.txt", delimiter=",")
########################################################################
########################################################################
#This is for the petitRADTRANS


pressures = np.logspace(-10, np.log10(P0_Ref), 100)



temperature = T_Ref*np.ones_like(pressures)
mass_fractions = {}

print("The molecular mass of the planet1 is:", Planet1.mu)

Mass_H2O_Ref = H2O_Ref*18.015/Planet1.mu
Mass_CO_Ref = CO_Ref*28.01/Planet1.mu
Mass_H2_Ref = H2_Ref*2.016/Planet1.mu
Mass_He_Ref = He_Ref*4.00/Planet1.mu

mass_fractions['H2'] = Mass_H2_Ref * np.ones_like(temperature)
#mass_fractions['He'] = Mass_He_Ref * np.ones_like(temperature)
#mass_fractions['H2O'] = Mass_H2O_Ref * np.ones_like(temperature)
#mass_fractions['CO'] = Mass_CO_Ref * np.ones_like(temperature)

MMW = Planet1.mu* np.ones_like(temperature)


#atmosphere1 = Radtrans(line_species = ['H2'], rayleigh_species = ['H2'],
#                        wlen_bords_micron = [0.3, 5.2])

atmosphereAll = Radtrans(line_species = ['H2'], rayleigh_species = ['H2'], \
                        continuum_opacities = ['H2-H2'], wlen_bords_micron = [0.3, 5.3], mode='lbl')

atmosphereRayleigh = Radtrans(rayleigh_species = ['H2'], wlen_bords_micron = [0.3, 5.3])
atmosphereCIA = Radtrans(continuum_opacities = ['H2-H2'], wlen_bords_micron = [0.3, 5.3])

atmosphereAll.setup_opa_structure(pressures)
atmosphereRayleigh.setup_opa_structure(pressures)
atmosphereCIA.setup_opa_structure(pressures)

print("The gravity of planet 1 is::", Planet1.Gp)

gravity = Planet1.Gp
R_pl = Planet1.Rp


P0 = P0_Ref

print("Calculating the transmission spectrum in petitRadtrans")

atmosphereAll.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)
atmosphereRayleigh.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)
atmosphereCIA.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

#####################################################################################################
#####################################################################################################
XValueAll = nc.c/atmosphereAll.freq/1e-4
YValueAll = (atmosphereAll.transm_rad/Planet1.Rs)**2
YValueRayleigh = (atmosphereRayleigh.transm_rad/Planet1.Rs)**2
YValueCIA = (atmosphereCIA.transm_rad/Planet1.Rs)**2

XValueTierra = Planet1.WavelengthArray*1e4
YValueTierra = T1.Spectrum
YValueTierraFalse = T1_False.Spectrum


fig, ax = plt.subplots(figsize=(12,10),nrows=1,ncols=1, sharex=True)
ax.plot(XValueTierra, YValueTierra, "k-", label="tierra")
ax.plot(XValueTierra, YValueTierraFalse, "g:", label="tierra_False")
ax.plot(nc.c/atmosphereAll.freq/1e-4, YValueAll, "r-", label="All Combined")
ax.plot(nc.c/atmosphereRayleigh.freq/1e-4, YValueRayleigh, "g-", label="Rayleigh")
ax.plot(nc.c/atmosphereCIA.freq/1e-4, YValueCIA, "b-", label="CIA")
ax.plot(MATLAB_Lam*1e4, MATLAB_Spectrum, "y-", label="MATLAB")
ax.set_xlim(0.3, 11.2)
ax.legend()
ax.set_xlabel("Wavelength (microns)")
ax.set_ylabel("Transit Depth (percent)")
fig.subplots_adjust(hspace=0.0)
plt.tight_layout()
SaveName = "/home/prajwal/Desktop/FeatureComparison_{}_H2_lbl_CIA.png".format(MoleculeSave)
plt.savefig(SaveName)
plt.close()



##Adding error to the observation
Wavelength1, WavelengthLower1, WavelengthUpper1, InterpolatedModel1, Noise1 = BinningDataCombined(WavelengthHS=XValue1,  RValue=100, ValuesHS=YValue1, ErrorFlag=False)
Wavelength2, WavelengthLower2, WavelengthUpper2, InterpolatedModel2, Noise2 = BinningDataCombined(WavelengthHS=XValue2,  RValue=100, ValuesHS=YValue2, ErrorFlag=False)




Offset = np.nanmedian(InterpolatedModel1) - np.nanmedian(InterpolatedModel2)
print("The Offset is ...", Offset)




plt.figure()
plt.plot(Wavelength1, InterpolatedModel1, "k-", label="petitRADTRANS")
plt.plot(Wavelength2, InterpolatedModel2, "r-", label="tierra cross-section")
plt.legend(loc=1)
plt.show()


#fig, ax = plt.subplots(figsize=(12,8),nrows=2,ncols=1, sharex=True)
#ax[0].plot(Wavelength1, InterpolatedModel1, "k-", label="petitRADTRANS")
#ax[0].plot(Wavelength2, InterpolatedModel2, "r-", label="tierra")
#ax[0].set_xlim(0.6, 11.3)
#ax[0].legend()
#ax[0].set_ylabel("(Rp/Rs)^2")
#ax[0].set_xlabel("Wavelength (microns)")
#ax[1].plot(Wavelength1, (InterpolatedModel1-InterpolatedModel2)/InterpolatedModel2*100, "k-", label="everything")
#ax[1].set_xlabel("Wavelength (microns)")
#ax[1].set_ylabel("Difference in percent")
#fig.subplots_adjust(hspace=0.0)
#plt.tight_layout()
#plt.suptitle(MoleculeSave)
#SaveName = "/home/prajwal/Desktop/FeatureComparison_{}_Resolution100000_H2_lbl_finesteps.png".format(MoleculeSave)
#plt.savefig(SaveName)
#plt.show(SaveName)
#plt.close()
