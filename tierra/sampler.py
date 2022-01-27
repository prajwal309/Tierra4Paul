import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from tierra import Target
from tierra.transmission import TransmissionSpectroscopy
from bisect import bisect
from multiprocessing import Pool
import os


def logLikelihood(theta, Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr):
    '''
    The log likelihood for calculation
    '''

    P0, T0, ALR, TInf, MR_N2Log, MR_COLog, MR_H2OLog, \
    MR_CO2Log, MR_CH4Log, MR_O3Log, MR_H2Log = theta

    if MR_N2Log>0.0 or MR_COLog>0.0 or MR_H2OLog>0.0 or MR_CO2Log>0.0 or MR_CH4Log>0.0 or \
    MR_O3Log>0.0 or MR_H2Log>0.0:
        return -np.inf

    if MR_N2Log<-20.0 or MR_COLog<-20.0 or MR_H2OLog<-20.0 or MR_CO2Log<-20.0 or MR_CH4Log<-20.0 or \
    MR_O3Log<-20.0 or MR_H2Log<-20.0:
        return -np.inf

    if TInf<100 or TInf>810:
        return -np.inf

    if T0<100 or T0>810:
        return -np.inf

    if P0<0 or P0>100:
        return -np.inf

    if np.sign(TInf-T0) == np.sign(ALR):
        return -np.inf

    #Converting the log value.
    MR_H2O = 10**MR_H2OLog
    MR_CO2 = 10**MR_CO2Log
    MR_CO = 10**MR_COLog
    MR_O3 = 10**MR_O3Log
    MR_CH4 = 10**MR_CH4Log
    MR_N2 = 10**MR_N2Log
    MR_H2 = 10**MR_H2Log

    MR_Combined = MR_H2O + MR_N2 + MR_CO + MR_CO2 + MR_CH4 + MR_O3 +\
                  MR_H2Log

    if MR_Combined>1.0:
        return -np.inf

    #Calculate the PT Profile
    CurrentSystem.PlanetParams['P0'] = P0
    CurrentSystem.PlanetParams['T0'] = T0
    CurrentSystem.PlanetParams['ALR'] = ALR
    CurrentSystem.PlanetParams['TInf'] = TInf
    CurrentSystem.PlanetParams['MR_H2O'] = MR_H2O
    CurrentSystem.PlanetParams['MR_CO2'] = MR_CO2
    CurrentSystem.PlanetParams['MR_CO'] = MR_CO
    CurrentSystem.PlanetParams['MR_O3'] = MR_O3
    CurrentSystem.PlanetParams['MR_CH4'] = MR_CH4
    CurrentSystem.PlanetParams['MR_N2'] = MR_N2
    CurrentSystem.PlanetParams['MR_H2'] = MR_H2



    global LeastResidual, ParameterNames, CurrentSaveName, CurrentzStep

    try:
        CurrentSystem.InitiateSystem()
        CurrentSystem.PT_Profile(zStep=CurrentzStep, ShowPlot=False)
        T1 = TransmissionSpectroscopy(CurrentSystem)
        T1.CalculateTransmission(CurrentSystem)
    except:
        print("Error for the following set of parameters.")
        for key,value in zip(ParameterNames, theta):
             print(key,":",value)
        return -np.inf

    CurrentWavelength = CurrentSystem.WavelengthArray*1e4
    CurrentModel = T1.Spectrum*1e6

    BinnedModel = np.zeros(len(Wavelength))

    global StartIndexAll, StopIndexAll
    counter = 0
    for StartIndex,StopIndex in zip(StartIndexAll, StopIndexAll):
        BinnedModel[counter] = np.mean(CurrentModel[StartIndex:StopIndex])
        counter+=1

    Residual = np.sum(np.power(Spectrum-BinnedModel,2)/(SpectrumErr*SpectrumErr))
    ChiSqr = -0.5*Residual


    if Residual<LeastResidual:
       print("Saving the best model.")
       LeastResidual = Residual
       with open("MCMCParams/BestParam"+CurrentSaveName+".txt", 'w+') as f:
          f.write("Residual:"+str(Residual)+"\n")
          for key,value in zip(ParameterNames, theta):
              f.write(key+":"+str(value)+"\n")
    return ChiSqr


def RunMCMC( PlanetParamsDict, StellarParamsDict, CSLocation=None, AssignedzStep=0.25, SubFolderName="CS_1", SaveName="Default", NSteps=1500, NCORES=4):
    '''
    Run MCMC value.

    Parameters
    ##########

    PlanetParamDict: dictionary
                     Dictionary containing planetary parameter value

    CSLocation: string
                Base location of the cross-section

    AssignedzStep: float
                    Assigned value for the zStep size. Should be smaller than 0.15

    StellarParamDict: dictionary
                      Dictionary containing stellar parameter value

    NumberPTLayers: integer
                    Number of PT layers for the calculation

    NSteps: integer
            Number of steps
    '''

    os.environ["OMP_NUM_THREADS"] = str(NCORES)

    #Load the data
    PlanetaryParameter = {}

    global zStep, LeastResidual, CurrentSaveName, CurrentzStep
    CurrentSaveName = SaveName
    LeastResidual = np.inf
    CurrentzStep = AssignedzStep

    if CSLocation:
        BaseLocation = CSLocation
    else:
        print("Using R1000 cross-section")
        BaseLocation="/media/prajwal/a66433b1-e5b2-467e-8ebf-5857f498dfce/LowerResolutionData/R1000"
        input("Would you like to proceed")

    global CurrentSystem
    CurrentSystem = Target.System(PlanetParamsDict, StellarParamsDict, LoadFromFile=False)
    CurrentSystem.LoadCrossSection(BaseLocation, SubFolder=SubFolderName)

    Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr  = np.loadtxt("data/Case1.R100.Earth.txt", delimiter=",", unpack=True)

    global StartIndexAll, StopIndexAll
    StartIndexAll = []
    StopIndexAll = []
    CurrentWavelength = CurrentSystem.WavelengthArray*1e4
    for Wl, Wp in zip(WavelengthLower, WavelengthUpper):
        StartIndex = bisect(CurrentWavelength, Wl)
        StopIndex = bisect(CurrentWavelength, Wp)
        StartIndexAll.append(StartIndex)
        StopIndexAll.append(StopIndex)

    nWalkers = 22
    ActualValue = []


    MR_H2OLog = np.log10(PlanetParamsDict['MR_H2O'])
    MR_CO2Log = np.log10(PlanetParamsDict['MR_CO2'])
    MR_COLog = np.log10(PlanetParamsDict['MR_CO'])
    MR_O3Log = np.log10(PlanetParamsDict['MR_O3'])
    MR_CH4Log = np.log10(PlanetParamsDict['MR_CH4'])
    MR_N2Log = np.log10(PlanetParamsDict['MR_N2'])
    MR_H2Log = np.log10(PlanetParamsDict['MR_H2'])

    #Convert the mixing ratio into log of the mixing ratio
    if np.isfinite(MR_H2OLog):
        MR_H2OLogErr = np.abs(0.2*MR_H2OLog)
    else:
        MR_H2OLog = -5
        MR_H2OLogErr = 0.25

    if np.isfinite(MR_CO2Log):
        MR_CO2LogErr = np.abs(0.2*MR_CO2Log)
    else:
        MR_CO2Log = -5
        MR_CO2LogErr = 0.25

    if np.isfinite(MR_COLog):
        MR_COLogErr = np.abs(0.2*MR_COLog)
    else:
        MR_COLog = -5
        MR_COLogErr = 0.25

    if np.isfinite(MR_O3Log):
        MR_O3LogErr = np.abs(0.2*MR_O3Log)
    else:
        MR_O3Log = -5
        MR_O3LogErr = 0.25

    if np.isfinite(MR_CH4Log):
        MR_CH4LogErr = np.abs(0.2*MR_CH4Log)
    else:
        MR_CH4Log = -5
        MR_CH4LogErr = 0.25

    if np.isfinite(MR_N2Log):
        MR_N2LogErr = np.abs(0.2*MR_N2Log)
    else:
        MR_N2Log = -5
        MR_N2LogErr = 0.25

    if np.isfinite(MR_H2Log):
        MR_H2LogErr = np.abs(0.2*MR_H2Log)
    else:
        MR_H2Log = -5
        MR_H2LogErr = 0.25

    P0Init = np.random.normal(PlanetParamsDict['P0'],0.1, nWalkers)            #Pressure at R_p in atm
    T0Init = np.random.normal(PlanetParamsDict['T0'], 10., nWalkers)            #Temperature at Rp
    ALRInit = np.random.normal(PlanetParamsDict['ALR'], 0.1, nWalkers)          #Adiabatic Lapse Rate in [K.km^{-1}]
    TInfInit = np.random.normal(PlanetParamsDict['TInf'], 10., nWalkers)          #Temperature in space in [K]:
    MR_N2LogInit = np.random.normal(MR_N2Log, MR_N2LogErr, nWalkers)      #Mixing ratio for nitrogen
    MR_COLogInit = np.random.normal(MR_COLog, MR_COLogErr, nWalkers)        #Mixing ratio for carbonmonoxide
    MR_H2OLogInit = np.random.normal(MR_H2OLog, MR_H2OLogErr, nWalkers)     #Mixing ratio for water
    MR_CO2LogInit = np.random.normal(MR_CO2Log, MR_CO2LogErr, nWalkers)     #Mixing ratio for carbondioxide
    MR_CH4LogInit = np.random.normal(MR_CH4Log, MR_CH4LogErr, nWalkers)     #Mixing ratio for methane
    MR_O3LogInit = np.random.normal(MR_O3Log, MR_O3LogErr, nWalkers)      #Mixing ratio for oxygen
    MR_H2LogInit = np.random.normal(MR_H2Log, MR_H2LogErr, nWalkers)          #Mixing ratio for hydrogen

    global ParameterNames
    ParameterNames = ["P0", "T0", "ALR", "TInf", "MR_N2Log", "MR_COLog", \
    "MR_H2OLog", "MR_CO2Log",  "MR_CH4Log", "MR_O3Log", "MR_H2Log"]



    StartingGuess = np.column_stack((P0Init, T0Init, ALRInit, TInfInit, \
                                     MR_N2LogInit, MR_COLogInit, MR_H2OLogInit, MR_CO2LogInit, \
                                     MR_CH4LogInit, MR_O3LogInit, MR_H2LogInit))

    _, nDim = np.shape(StartingGuess)

    #with Pool() as pool:
    #    sampler = emcee.EnsembleSampler(nWalkers, nDim, logLikelihood, args=[Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr], pool=pool)
    #    sampler.run_mcmc(StartingGuess, NSteps, progress=True)

    sampler = emcee.EnsembleSampler(nWalkers, nDim, logLikelihood, args=[Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr])
    sampler.run_mcmc(StartingGuess, NSteps, progress=True)

    #Make the best parameter
    LocX, LocY = np.where(np.max(sampler.lnprobability)==sampler.lnprobability)
    BestParameters = sampler.chain[LocX[0], LocY[0], :]

    BestP0, BestT0, BestALR, BestTInf, BestMR_N2Log, BestMR_COLog, BestMR_H2OLog, \
    BestMR_CO2Log, BestMR_CH4Log, BestMR_O3Log, BestMR_H2Log = BestParameters

    print("The best P0 value is::", BestP0)
    print("The best temperature is::", BestT0)

    #Converting the log value.
    MR_H2O = 10**BestMR_H2OLog
    MR_CO2 = 10**BestMR_CO2Log
    MR_CO = 10**BestMR_COLog
    MR_O3 = 10**BestMR_O3Log
    MR_CH4 = 10**BestMR_CH4Log
    MR_N2 = 10**BestMR_N2Log
    MR_H2 = 10**BestMR_H2Log

    #Calculate the PT Profile
    CurrentSystem.PlanetParams['P0'] = BestP0
    CurrentSystem.PlanetParams['T0'] = BestT0
    CurrentSystem.PlanetParams['ALR'] = BestALR
    CurrentSystem.PlanetParams['TInf'] = BestTInf
    CurrentSystem.PlanetParams['MR_H2O'] = MR_H2O
    CurrentSystem.PlanetParams['MR_CO2'] = MR_CO2
    CurrentSystem.PlanetParams['MR_CO'] = MR_CO
    CurrentSystem.PlanetParams['MR_O3'] = MR_O3
    CurrentSystem.PlanetParams['MR_CH4'] = MR_CH4
    CurrentSystem.PlanetParams['MR_N2'] = MR_N2
    CurrentSystem.PlanetParams['MR_H2'] = MR_H2

    CurrentSystem.InitiateSystem()
    CurrentSystem.PT_Profile(zStep=CurrentzStep, ShowPlot=False)
    T1 = TransmissionSpectroscopy(CurrentSystem)
    T1.CalculateTransmission(CurrentSystem)

    plt.figure(figsize=(12,8))
    plt.plot(-np.mean(sampler.lnprobability, axis=0))
    plt.yscale("log")
    plt.savefig("Figures/LogProbability_"+SaveName+".png")
    plt.close('all')


    plt.figure(figsize=(12,8))
    plt.errorbar(Wavelength, Spectrum, yerr=SpectrumErr, capsize=4, color="green", linestyle="None")
    plt.plot(CurrentSystem.WavelengthArray*1e4, T1.Spectrum*1e6, "r-", label="Best Model")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$(R_p/R_s)^2$")
    plt.xlim(min(Wavelength)-0.1, max(Wavelength)+0.1)
    plt.ylim(min(T1.Spectrum*1e6), max(T1.Spectrum*1e6))
    plt.savefig("Figures/BestModel_"+SaveName+".png")
    plt.close('all')

    #Now use the data
    Samples = sampler.chain
    X,Y,Z = np.shape(Samples)
    print(X,Y,Z)

    SaveMCMCName = "MCMC_Data/"+SaveName+".npy"
    np.save(SaveMCMCName, Samples)

    #Remove the burnin
    SamplesRemoved = Samples[:,Y//2:,:]
    SamplesFlattened = SamplesRemoved.reshape(X*Y//2, Z)

    SaveFigName = "Figures/"+SaveName+".png"
    plt.figure(figsize=(20,20))
    corner.corner(SamplesFlattened, labels=ParameterNames, title_fmt="5.3f",quantiles=[0.158, 0.5, 0.842], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(SaveFigName)
    plt.close()

    #Now save the best figure
