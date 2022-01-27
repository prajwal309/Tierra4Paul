import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from tierra import Target
from tierra.transmission import TransmissionSpectroscopy
from bisect import bisect
from multiprocessing import Pool
import os
from tierra.JWSTErrorbar import BinningDataNIRSpecPrism, BinningDataCombined

import glob
import datetime
import h5py


def SaveProgress():

    global sampler, StartTime, CurrentSaveName1, Iteration, nWalkers

    #Save the log probability
    LogProbability = sampler.lnprobability
    Samples = sampler.chain
    CurrentStep = int(round(Iteration/(nWalkers)))

    print("Now saving the file at step size of:", CurrentStep)
    h5pySaveName = "ProgressData/"+CurrentSaveName+"_"+StartTime+"_"+str(CurrentStep).zfill(6)+".h5py"

    if not(os.path.exists("ProgressData")):
        os.system("mkdir ProgressData")

    with h5py.File(h5pySaveName, 'w') as f:
        f['lnprobabity'] = sampler.lnprobability
        f['chain'] = sampler.chain


    #If more than three files, keep the last two.
    AllFileName = np.array(glob.glob("ProgressData/{}_{}_*.h5py".format(CurrentSaveName, StartTime)))

    if len(AllFileName)>2:
        Time = np.array([int(Item.split("_")[-2][:-5].replace("-","")) for Item in AllFileName])
        FileStep = np.array([int(Item.split("_")[-1][:-5]) for Item in AllFileName])
        ArrangeIndex = np.argsort(FileStep)
        AllFileName = AllFileName[ArrangeIndex]

        #remove everything except the last two
        for FileItem in AllFileName[:-2]:
            os.system("rm {}".format(FileItem))


def logLikelihood(theta, Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr):
    '''
    The log likelihood for calculation
    '''

    global LeastResidual, ParameterNames, CurrentSaveName, \
           Iteration, nWalkers, CurrentSaveInterval

    #Save before the priors
    if (Iteration+1)%(CurrentSaveInterval*nWalkers)==0:
        SaveProgress()

    T0, LogALR, TInf, N0_N2Log, N0_COLog, N0_H2OLog, \
    N0_CO2Log, N0_CH4Log, N0_O3Log, N0_H2Log  = theta

    if min([N0_N2Log, N0_COLog, N0_H2OLog, N0_CO2Log, N0_CH4Log, N0_O3Log, N0_H2Log])<0.0:
        print("min case")
        return -np.inf

    if max([N0_N2Log, N0_COLog, N0_H2OLog, N0_CO2Log, N0_CH4Log, N0_O3Log, N0_H2Log])>50.0:
        print("max case")
        return -np.inf

    if TInf<100 or TInf>810:
        print("TInf case")
        return -np.inf

    if T0<100 or T0>810:
        print("T0 case")
        return -np.inf

    if LogALR>1 or LogALR<-15:
        print("ALR Case")
        return -np.inf

    #Converting the log value.
    N0_N2 = 10**N0_N2Log
    N0_CO = 10**N0_COLog
    N0_H2O = 10**N0_H2OLog
    N0_CO2 = 10**N0_CO2Log
    N0_O3 = 10**N0_O3Log
    N0_CH4 = 10**N0_CH4Log
    N0_H2 = 10**N0_H2Log
    N0_He = N0_H2*15./85.

    Total_N0 = N0_N2 + N0_CO + N0_H2O + N0_CO2 + \
               N0_O3 + N0_CH4 + N0_H2 + N0_He

    #Calculate the total pressure
    P0 = Total_N0/6.023e23*T0/273.15

    if P0>100 or P0<1e-4:
        return -np.inf

    #Calculating the mixing ratio
    MR_N2 = N0_N2/Total_N0
    MR_CO = N0_CO/Total_N0
    MR_H2O = N0_H2O/Total_N0
    MR_CO2 = N0_CO2/Total_N0
    MR_O3 = N0_O3/Total_N0
    MR_CH4 = N0_CH4/Total_N0
    MR_H2 = N0_H2/Total_N0

    #Calculate the PT Profile
    CurrentSystem.PlanetParams['P0'] = P0
    CurrentSystem.PlanetParams['T0'] = T0
    CurrentSystem.PlanetParams['ALR'] = 10**LogALR
    CurrentSystem.PlanetParams['TInf'] = TInf
    CurrentSystem.PlanetParams['MR_H2O'] = MR_H2O
    CurrentSystem.PlanetParams['MR_CO2'] = MR_CO2
    CurrentSystem.PlanetParams['MR_CO'] = MR_CO
    CurrentSystem.PlanetParams['MR_O3'] = MR_O3
    CurrentSystem.PlanetParams['MR_CH4'] = MR_CH4
    CurrentSystem.PlanetParams['MR_N2'] = MR_N2
    CurrentSystem.PlanetParams['MR_H2'] = MR_H2

    CurrentSystem.InitiateSystem()
    CurrentSystem.PT_Profile(zStep=0.25, ShowPlot=False)
    T1 = TransmissionSpectroscopy(CurrentSystem, CIA=True)
    T1.CalculateTransmission(CurrentSystem)

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
       #print("Saving the best model.")
       LeastResidual = Residual
       with open("MCMCParams/BestParam"+CurrentSaveName+".txt", 'w+') as f:
          f.write("Residual:"+str(Residual)+"\n")
          for key,value in zip(ParameterNames, theta):
              f.write(key+":"+str(value)+"\n")
       if 1==1:

          plt.figure(figsize=(12,8))
          plt.subplot(211)
          plt.errorbar(Wavelength, Spectrum, yerr=SpectrumErr, linestyle='None')
          plt.plot(Wavelength, BinnedModel, "r-")
          plt.xlim([min(Wavelength),max(Wavelength)])
          plt.ylabel("(Rp/Rs)^2")
          plt.subplot(212)
          plt.plot(Wavelength, (Spectrum-BinnedModel)/SpectrumErr, "ko")
          plt.xlim([min(Wavelength),max(Wavelength)])
          plt.xlabel("Wavelength (Microns)")
          plt.ylabel("Deviation")
          plt.tight_layout()
          plt.savefig("Figures/CurrentBestModel_%s.png" %CurrentSaveName)
          plt.close('all')
          print("Best Model Updated. Figure saved.")
    Iteration+=1
    return ChiSqr


def RunMCMC( PlanetParamsDict, StellarParamsDict, CSLocation=None, AssignedzStep=0.25, SubFolderName="CS_1", SaveName="Default", NSteps=5000, NCORES=4, NewStart=False, SaveInterval=25):
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

    #Initiate the planetary parameters
    PlanetaryParameter = {}

    global LeastResidual, CurrentSaveName, CurrentzStep, StartTime
    StartTime = str(datetime.datetime.now()).replace(" ", "-").replace(":","-").split(".")[0]
    CurrentSaveName = SaveName
    LeastResidual = np.inf
    CurrentzStep = AssignedzStep

    if CSLocation:
        BaseLocation = CSLocation
    else:
        print("Using R1000 cross-section")
        BaseLocation="/media/prajwal/a66433b1-e5b2-467e-8ebf-5857f498dfce/LowerResolutionData/R1000"


    global CurrentSystem
    CurrentSystem = Target.System(PlanetParamsDict, StellarParamsDict, LoadFromFile=False)
    CurrentSystem.LoadCrossSection(BaseLocation, SubFolder=SubFolderName, CIA=True)

    if "HJ" in CurrentSaveName:
        WavelengthLower, WavelengthUpper, Wavelength, Spectrum, SpectrumErr  = np.loadtxt("FittingData/WarmJupiter_FIT.data", delimiter=",", unpack=True, skiprows=1)
    elif "SE" in CurrentSaveName:
        WavelengthLower, WavelengthUpper, Wavelength, Spectrum, SpectrumErr  = np.loadtxt("FittingData/SuperEarth_FIT.data", delimiter=",", unpack=True, skiprows=1)
    else:
        print("Error in the type of file")
        assert 1==2


    global StartIndexAll, StopIndexAll, nWalkers, CurrentSaveInterval, Iteration

    Iteration = 0

    CurrentSaveInterval = SaveInterval
    print("The current saving interval is::", CurrentSaveInterval)

    StartIndexAll = []
    StopIndexAll = []
    CurrentWavelength = CurrentSystem.WavelengthArray*1e4
    for Wl, Wp in zip(WavelengthLower, WavelengthUpper):
        StartIndex = bisect(CurrentWavelength, Wl)
        StopIndex = bisect(CurrentWavelength, Wp)
        StartIndexAll.append(StartIndex)
        StopIndexAll.append(StopIndex)

    nWalkers = 20
    ActualValue = []

    TotalNumbers = 6.023e23*PlanetParamsDict['P0']*273.15/PlanetParamsDict['T0']


    N0_N2Log = np.log10(PlanetParamsDict['MR_N2']*TotalNumbers)
    N0_COLog = np.log10(PlanetParamsDict['MR_CO']*TotalNumbers)
    N0_H2OLog = np.log10(PlanetParamsDict['MR_H2O']*TotalNumbers)
    N0_CO2Log = np.log10(PlanetParamsDict['MR_CO2']*TotalNumbers)
    N0_CH4Log = np.log10(PlanetParamsDict['MR_CH4']*TotalNumbers)
    N0_O3Log = np.log10(PlanetParamsDict['MR_O3']*TotalNumbers)
    N0_H2Log = np.log10(PlanetParamsDict['MR_H2']*TotalNumbers)


    CalcP0 = TotalNumbers/6.023e23*PlanetParamsDict['T0']/273.15
    print("The calculated pressure is given:", CalcP0)

    FileName = glob.glob("ProgressData/{}*.h5py".format(SaveName))
    FileExist = len(FileName)>0

    #Convert the mixing ratio into log of the mixing ratio
    if NewStart or not(FileExist):
        print("Initializing from the random")
        T0Init = np.random.normal(PlanetParamsDict['T0'], 20., nWalkers)                      #Temperature at Rp
        ALRInit = np.random.normal(np.log10(PlanetParamsDict['ALR']), 0.25, nWalkers)         #Adiabatic Lapse Rate in [K.km^{-1}]
        TInfInit = np.random.normal(PlanetParamsDict['TInf'], 20., nWalkers)                   #Temperature in space in [K]:

        N0_N2LogInit = np.random.normal(N0_N2Log, 0.5, nWalkers)                              #Mixing ratio for nitrogen
        N0_COLogInit = np.random.normal(N0_COLog, 0.5, nWalkers)                              #Mixing ratio for carbonmonoxide
        N0_H2OLogInit = np.random.normal(N0_H2OLog, 0.5, nWalkers)                            #Mixing ratio for water
        N0_CO2LogInit = np.random.normal(N0_CO2Log, 0.5, nWalkers)                            #Mixing ratio for carbondioxide
        N0_CH4LogInit = np.random.normal(N0_CH4Log, 0.5, nWalkers)                            #Mixing ratio for methane
        N0_O3LogInit = np.random.normal(N0_O3Log, 0.5, nWalkers)                              #Mixing ratio for oxygen
        N0_H2LogInit = np.random.normal(N0_H2Log, 0.5, nWalkers)
                                      #Mixing ratio for oxygen
        StartingGuess = np.column_stack((T0Init, ALRInit, TInfInit, \
                                        N0_N2LogInit, N0_COLogInit, N0_H2OLogInit,
                                        N0_CO2LogInit, N0_CH4LogInit, N0_O3LogInit,
                                        N0_H2LogInit))

    else:
        FileName = np.array(FileName)

        Time = np.array([int(Item.split("_")[-2][:-5].replace("-","")) for Item in FileName])
        FileStep = np.array([int(Item.split("_")[-1][:-5]) for Item in FileName])
        CommonIndex = Time*1e6+FileStep
        SelectIndex = np.argmax(CommonIndex)
        SelectedFile = FileName[SelectIndex]

        print("Loading the parameters from::", SelectedFile)

        Data = h5py.File(SelectedFile, 'r')

        Data = h5py.File(SelectedFile, 'r')
        #Check with h5py.File(h5pySaveName, 'w') as f:
        CurrentLogProbablity = np.abs(np.mean(Data['lnprobabity'], axis=0))
        SelectIndex = CurrentLogProbablity>1

        CurrentLogProbablity = CurrentLogProbablity[SelectIndex]
        CurrentChain = Data['chain']
        BestIndex = np.argmin(CurrentLogProbablity)
        StartingGuess = Data['chain'][:,BestIndex,:]

        M, N = np.shape(StartingGuess)
        ReArrangeIndex = np.arange(M)
        np.random.shuffle(ReArrangeIndex)

        StartingGuess = StartingGuess[ReArrangeIndex, :]

    global ParameterNames, sampler, I
    ParameterNames = ["T0", "Log_ALR", "TInf", "N0_N2Log", "N0_COLog", \
    "N0_H2OLog", "N0_CO2Log",  "N0_CH4Log", "N0_O3Log", "N0_H2Log"]

    _, nDim = np.shape(StartingGuess)

    sampler = emcee.EnsembleSampler(nWalkers, nDim, logLikelihood, args=[Wavelength, WavelengthLower, WavelengthUpper, Spectrum, SpectrumErr])
    #sampler.run_mcmc(StartingGuess, NSteps, progress=True)
    sampler.run_mcmc(StartingGuess, NSteps)

    try:
        print("Printing the auto-correlation time.")
        print(sampler.get_autocorr_time())
    except:
        print("The chains have not converged.")
        pass

    #Make the best parameter
    LocX, LocY = np.where(np.max(sampler.lnprobability)==sampler.lnprobability)
    BestParameters = sampler.chain[LocX[0], LocY[0], :]

    BestT0, BestALR, BestTInf, BestN0_N2Log, BestN0_COLog, BestN0_H2OLog, \
    BestN0_CO2Log, BestN0_CH4Log, BestN0_O3Log, BestN0_H2Log = BestParameters


    #Converting the log value.
    N0_N2 = 10**BestN0_N2Log
    N0_CO = 10**BestN0_COLog
    N0_H2O = 10**BestN0_H2OLog
    N0_CO2 = 10**BestN0_CO2Log
    N0_O3 = 10**BestN0_O3Log
    N0_CH4 = 10**BestN0_CH4Log
    N0_H2 = 10**BestN0_H2Log
    N0_He = 15./85.*N0_H2

    Total_N0 = N0_He+N0_H2+N0_N2+N0_CO+N0_H2O+N0_CO2+N0_O3+N0_CH4

    #Calculating the mixing ratio
    MR_H2O = N0_H2O/Total_N0
    MR_CO2 = N0_CO2/Total_N0
    MR_CO = N0_CO/Total_N0
    MR_O3 = N0_O3/Total_N0
    MR_CH4 = N0_CH4/Total_N0
    MR_N2 = N0_N2/Total_N0
    MR_H2 = N0_H2/Total_N0

    BestP0 = Total_N0/6.023e23*BestT0/273.15

    print("The best P0 value is::", BestP0)
    print("The best temperature is::", BestT0)
    print("The best ALR is::", BestALR)

    #Calculate the PT Profile
    CurrentSystem.PlanetParams['P0'] = BestP0
    CurrentSystem.PlanetParams['T0'] = BestT0
    CurrentSystem.PlanetParams['ALR'] = 10**BestALR
    CurrentSystem.PlanetParams['TInf'] = BestTInf
    CurrentSystem.PlanetParams['MR_H2O'] = MR_H2O
    CurrentSystem.PlanetParams['MR_CO2'] = MR_CO2
    CurrentSystem.PlanetParams['MR_CO'] = MR_CO
    CurrentSystem.PlanetParams['MR_O3'] = MR_O3
    CurrentSystem.PlanetParams['MR_CH4'] = MR_CH4
    CurrentSystem.PlanetParams['MR_N2'] = MR_N2
    CurrentSystem.PlanetParams['MR_H2'] = MR_H2

    CurrentSystem.InitiateSystem()
    CurrentSystem.PT_Profile(zStep=0.25, ShowPlot=False)
    T1 = TransmissionSpectroscopy(CurrentSystem)
    T1.CalculateTransmission(CurrentSystem)

    plt.figure(figsize=(12,8))
    plt.plot(-np.mean(sampler.lnprobability, axis=0))
    plt.yscale("log")
    plt.savefig("Figures/LogProbability_"+SaveName+".png")
    plt.close('all')

    XValue = CurrentSystem.WavelengthArray*1e4
    YValue = T1.Spectrum*1e6

    WavelengthNew, _, _, ModelNew, Noise = BinningDataCombined(WavelengthHS=XValue,  RValue=100, ValuesHS=YValue, ErrorFlag=False)

    plt.figure(figsize=(12,8))
    plt.errorbar(Wavelength, Spectrum, yerr=SpectrumErr, capsize=4, color="green", linestyle="None", label="Data")
    #plt.plot(CurrentSystem.WavelengthArray*1e4, T1.Spectrum*1e6, "r-", label="Best Model")
    plt.plot(WavelengthNew, ModelNew, "r-", label="Best Model")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$(R_p/R_s)^2$")
    plt.xlim(min(Wavelength)-0.1, max(Wavelength)+0.1)
    plt.ylim(min(T1.Spectrum*1e6), max(T1.Spectrum*1e6))
    plt.savefig("Figures/BestModel_"+SaveName+".png")
    plt.close('all')

    #Now use the data
    Samples = sampler.chain
    X,Y,Z = np.shape(Samples)

    SaveMCMCName = "MCMC_Data/"+SaveName+".npy"
    np.save(SaveMCMCName, Samples)

    #Remove the burnin
    SamplesRemoved = Samples[:,Y//2:,:]
    SamplesFlattened = SamplesRemoved.reshape(X*Y//2, Z)

    SaveFigName = "Figures/Corner_"+SaveName+".png"
    plt.figure(figsize=(20,20))
    corner.corner(SamplesFlattened, labels=ParameterNames, title_fmt="5.3f",quantiles=[0.158, 0.5, 0.842], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(SaveFigName)
    plt.close()

    #Now save the best figure
