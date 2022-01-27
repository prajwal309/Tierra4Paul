import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect

class JWSTInstrument:
    def __init__(self, Target, Resolution=100):
        '''
        mode, UserBins=None
        '''

        self.mode = mode
        if mode == "NIRSpec_prism":
            self.WL_Start = 600e-9
            self.WL_Final = 5200e-9
            self.Resolution = 200

            if UserBins is None:
                self.WaveLengthBins = self.GetWavelengthBins()
            else:
                self.WaveLengthBins = UserBins
        else:
            raise("No such modes were found")

    def GetWavelengthBins(self):
        WaveLengthValues = [self.WL_Start]
        while WaveLengthValues[-1]<self.WL_Final:
            WaveLengthValues.append(WaveLengthValues[-1]+WaveLengthValues[-1]/self.Resolution)
        return WaveLengthValues


    def EstimateJWSTError(self, Target, NumTransits=1, TDur=1.0):
        '''
        This method estimates the error
        in target.
        ================================================
        Parameters
        ================================================
        Target: Target is an TRAPPIST like object which has
        NumTransits:Number of transits observed.
        TDur:Transit duration in hours.
        ================================================
        '''

        NoiseFloor = 20.0                                   #ppm

        #High resolution wavelength and intensity from the blackbody ---> Can be replaced by stellar models
        WaveLength_HR, Intensity_HR = Target.BlackbodyEmission()

        #Get the relative Intensity
        #J band goes from 1.1 to 1.4 microns
        print("The magnitude of the target is::", Target.Magnitude)

        #Reference from https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
        #For J band
        dl_l = 0.16
        m0_Flux = 1600

        #zero magninitude

        NPhotons = 1600.0/10**(Target.Magnitude/2.5)*1.51e7*dl_l*1e-4

        #JWST parameters
        Area =  25.3*1e4   #JWST effective area in cm^2


        #Calculate the expected number of photons in J band
        Qe = 0.95


        #JBand_PhotonCount = Flux/EnergyPhoton*Qe*NumTransits*TDur*3600*Area
        JBand_PhotonCount = NPhotons*Qe*NumTransits*TDur*3600*Area
        DetectorMultiplier = 1.0
        print("JBand_PhotonCount::", JBand_PhotonCount)

        #Dark Current Rate
        Dark_Current_Rate

        #Now area under curve
        StartIndex = bisect(WaveLength_HR, 1.1e-6)
        StopIndex = bisect(WaveLength_HR, 1.4e-6)

        NormArea= np.trapz(Intensity_HR[StartIndex:StopIndex],WaveLength_HR[StartIndex:StopIndex])


        Photons_Wavelength = np.zeros(len(self.WaveLengthBins))

        assert(len(self.WaveLengthBins)>3)
        StartWaveLength = 2*self.WaveLengthBins[0]-self.WaveLengthBins[1]

        for i in range(len(Photons_Wavelength)):
            StartIndex = bisect(WaveLength_HR,StartWaveLength)
            StopIndex = bisect(WaveLength_HR,self.WaveLengthBins[i])
            CurrentArea = np.trapz(Intensity_HR[StartIndex:StopIndex],WaveLength_HR[StartIndex:StopIndex])
            NormalizationFactor = CurrentArea/NormArea
            Photons_Wavelength[i] = DetectorMultiplier*JBand_PhotonCount*NormalizationFactor
            StartWaveLength = self.WaveLengthBins[i]

        PoissonNoise = np.sqrt(Photons_Wavelength)
        print("The number of bins is::", 1500.0/len(Photons_Wavelength))

        ReadNoise_PerBin = 1500.0/len(Photons_Wavelength)*7.0    #10 electrons per pixel
        OtherNoise_PerBin = 1500.0/len(Photons_Wavelength)*7.0   #15 electrons per pixel

        SNR = Photons_Wavelength/np.sqrt(2.0*Photons_Wavelength)#+ReadNoise_PerBin**2+OtherNoise_PerBin**2)
        #Return SNR in ppm

        return SNR



    def BinFluxModel(self,nu_HR,sigma_HR):
        '''
        This function takes a cross-section at high resolution:
        nu_HR is the wavenumber in increasing order
        abs_HR is the absorption cross-section in an increasing order
        The stepsize in the WaveNumberGrid is not expected to be the equal
        '''

        nu_Grid = self.WaveLengthBins
        InterpValues = np.zeros(len(self.WaveLengthBins))
        StartValue = 0

        i = 0           #Counter
        while i<len(nu_Grid):
            StartIndex = bisect(nu_HR, StartValue)
            StopIndex = bisect(nu_HR, self.WaveLengthBins[i])
            print(i, StartIndex, StopIndex)
            InterpValues[i] = np.mean(sigma_HR[StartIndex:StopIndex])
            StartValue=self.WaveLengthBins[i]
            i+=1
        #Remove the nan index
        NanIndex = np.isnan(InterpValues)
        InterpValues[NanIndex] = 0.0
        return InterpValues



class Target:
    def __init__(self, Name, Magnitude, MStar, RStar, Temp):
        self.Name = Name
        self.Magnitude = Magnitude
        self.MStar = MStar
        self.RStar = RStar
        self.Temp = Temp

    def BlackbodyEmission(self):
        '''
        Parameters
        ===========================
        Gives the transmission spectrum in a wavelength window
        '''
        h = 6.626e-34
        c = 3e6
        k = 1.38e-23
        WaveLength = np.arange(300,35000,0.1)*1e-9

        a = 2.0*h*c**2
        print("The temperature of the star is::", self.Temp)
        b = h*c/(WaveLength*k*self.Temp)
        Intensity = a/((WaveLength**5)*(np.exp(b)-1.0))
        return WaveLength, Intensity


def UnitTest():
    Instrument = JWSTInstrument(mode="NIRSpec_prism")

    TRAPPIST = Target("TRAPPIST", 11.0, 0.1, 0.1, 2700)
    #Intensity = TRAPPIST.BlackbodyEmission()
    Instrument.EstimateJWSTError(TRAPPIST, NumTransits=1)


    NumberTransit = 1.0


def DefaultInstrumentSNR(ParamStellar, NumTransits=1.0, WaveLengthBins=None):
    """
    Parameter:
    ----------------------------------------------------------------------------
    ParamStellar: Dictionary parameters for the star containing J-Magnitude
    NumTransits: Number of transits planned for observation
    WaveLengthBins: User defined bins for the wavelength
    ----------------------------------------------------------------------------
    """
    Instrument = JWSTInstrument(mode="NIRSpec_prism",UserBins=WaveLengthBins)
    Target01 = Target("NoName", ParamStellar["JMag"], \
                        ParamStellar["Mass"], ParamStellar["Radius"],\
                        ParamStellar["TEff"])
    #Intensity = TRAPPIST.BlackbodyEmission()

    return Instrument.EstimateJWSTError(Target01, NumTransits=NumTransits)



        def EstimateJWSTError(self, Target, NumTransits=1, TDur=1.0):
            '''
            This method estimates the error
            in target.
            ================================================
            Parameters
            ===========
            Target: Tierra target object
                    Tierra object that has access to tierra

            NumTransits: Integer
                         Number of transits that will be stacked on top of one another

            TDur:Transit duration in hours
                 The transit duration for the transit

            '''

            NoiseFloor = 20.0                                   #ppm

            #High resolution wavelength and intensity from the blackbody ---> Can be replaced by stellar models
            WaveLength_HR, Intensity_HR = Target.BlackbodyEmission()

            #Get the relative Intensity
            #J band goes from 1.1 to 1.4 microns
            print("The magnitude of the target is::", Target.Magnitude)

            #Reference from https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
            #For J band
            dl_l = 0.16
            m0_Flux = 1600

            NPhotons = 1600.0/10**(Target.Magnitude/2.5)*1.51e7*dl_l*1e-4
            print("Method 2::", NPhotons)
            input("Wait here...")

            #JWST parameters
            Area =  25.3*1e4   #JWST effective area in cm^2

            #Calculate the expected number of photons in J band
            Qe = 0.95

            print("The number of transits is given by:", NumTransits)
            #JBand_PhotonCount = Flux/EnergyPhoton*Qe*NumTransits*TDur*3600*Area
            JBand_PhotonCount = NPhotons*Qe*NumTransits*TDur*3600*Area
            DetectorMultiplier = 1.0
            print("JBand_PhotonCount::", JBand_PhotonCount)

            #Dark Current Rate
            Dark_Current_Rate

            #Now area under curve
            StartIndex = bisect(WaveLength_HR, 1.1e-6)
            StopIndex = bisect(WaveLength_HR, 1.4e-6)

            NormArea= np.trapz(Intensity_HR[StartIndex:StopIndex],WaveLength_HR[StartIndex:StopIndex])


            Photons_Wavelength = np.zeros(len(self.WaveLengthBins))

            assert(len(self.WaveLengthBins)>3)
            StartWaveLength = 2*self.WaveLengthBins[0]-self.WaveLengthBins[1]

            for i in range(len(Photons_Wavelength)):
                StartIndex = bisect(WaveLength_HR,StartWaveLength)
                StopIndex = bisect(WaveLength_HR,self.WaveLengthBins[i])
                CurrentArea = np.trapz(Intensity_HR[StartIndex:StopIndex],WaveLength_HR[StartIndex:StopIndex])
                NormalizationFactor = CurrentArea/NormArea
                Photons_Wavelength[i] = DetectorMultiplier*JBand_PhotonCount*NormalizationFactor
                StartWaveLength = self.WaveLengthBins[i]

            print("The photons wavelength is given by:")
            print(Photons_Wavelength)

            PoissonNoise = np.sqrt(Photons_Wavelength)
            print("The number of bins is::", 1500.0/len(Photons_Wavelength))

            ReadNoise_PerBin = 1500.0/len(Photons_Wavelength)*7.0    #10 electrons per pixel
            OtherNoise_PerBin = 1500.0/len(Photons_Wavelength)*7.0   #15 electrons per pixel

            SNR = Photons_Wavelength/np.sqrt(2.0*Photons_Wavelength)#+ReadNoise_PerBin**2+OtherNoise_PerBin**2)
            #Return SNR in ppm

            return SNR

    def PlanckFunction(self):
        '''
        Calculates the Planck function in order to calculate
        '''
        pass




    def BinFluxModel(self,nu_HR,sigma_HR):
        '''
        This function takes a cross-section at high resolution:
        nu_HR is the wavenumber in increasing order
        abs_HR is the absorption cross-section in an increasing order
        The stepsize in the WaveNumberGrid is not expected to be the equal
        '''

        nu_Grid = self.WaveLengthBins
        InterpValues = np.zeros(len(self.WaveLengthBins))
        StartValue = 0

        i = 0           #Counter
        while i<len(nu_Grid):
            StartIndex = bisect(nu_HR, StartValue)
            StopIndex = bisect(nu_HR, self.WaveLengthBins[i])
            print(i, StartIndex, StopIndex)
            InterpValues[i] = np.mean(sigma_HR[StartIndex:StopIndex])
            StartValue=self.WaveLengthBins[i]
            i+=1
        #Remove the nan index
        NanIndex = np.isnan(InterpValues)
        InterpValues[NanIndex] = 0.0
        return InterpValues
