import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
import sys
from numba import prange


class TransmissionSpectroscopy:

    def __init__(self, Target, CIA=False):
        '''
        Initiate the transmission
        '''


        #Flag to collision induced absportion flag

        self.CIA = CIA

        sz = Target.NumLayers
        self.dz_cm= np.concatenate(([Target.zValuesCm[0]], np.diff(Target.zValuesCm)))


        Z_ii, Z_jj = np.meshgrid(Target.zValuesCm[1:], Target.zValuesCm[:-1])
        self.x__ = np.sqrt((Target.Rp+Z_ii)*(Target.Rp+Z_ii)
                   -(Target.Rp+Z_jj)*(Target.Rp+Z_jj))
        self.x__[np.isnan(self.x__)]=0.0


        x__Copy = np.copy(self.x__)
        self.xNew_ = np.pad(x__Copy,(1,0), mode='constant')[1:,:-1]
        self.ds_= self.x__- self.xNew_

        #Averaging the distance
        dsCopy = np.copy(self.ds_)
        self.dsNew_ = np.pad(dsCopy,(1,0),mode='constant')[1:,:-1]
        self.ds_ = 0.5*(self.ds_ + self.dsNew_)


        ###########################################################################
        ###########################################################################

       


    def CalculateContributionFunction(self, Target, ShowPlot=False, interpolation="bilinear"):
        '''
        This method calculates the spectrum given the

        Parameters:
        -----------
        Target: Tierra Target object

        ShowPlot: Boolean

        interpolation: string
            Either use the bilinear or hill method

        Returns
        --------
        Array

        Spectrum of the planet is returned.

        '''

        #Now solve for the atmosphere of the planet
        
        self.AllSpectrum = np.zeros((len(Target.WavelengthArray), Target.NumLayers+1), dtype=np.float64)
        
        
        #for self.CurrentLayer in prange(Target.NumLayers):
        for Outerlayer in range(-1,Target.NumLayers):
            print("The value of Outerlayer is:", Outerlayer)

            #Initialize the spectrum to zeros
            self.Spectrum = np.zeros(len(Target.WavelengthArray), dtype=np.float64)
            
            #Initializing the alpha function
            self.alpha = np.zeros((len(Target.WavelengthArray),Target.NumLayers),dtype=np.float64)
            for self.CurrentLayer in range(Target.NumLayers):

                if Outerlayer==self.CurrentLayer:
                    continue

                    
                CurrentT = Target.TzAnalytical[self.CurrentLayer]
                CurrentP = np.log10(Target.PzAnalytical[self.CurrentLayer])

                TIndex = bisect_left(Target.TemperatureArray, CurrentT)
                co_t = (CurrentT-Target.TemperatureArray[TIndex-1])/(Target.TemperatureArray[TIndex]-Target.TemperatureArray[TIndex-1])

                PIndex = bisect_left(Target.PressureArray, CurrentP)
                #Add if statement for equality

                #Use different interpolation method

                if "bilinear" in interpolation.lower():
                    co_t = (CurrentT-Target.TemperatureArray[TIndex-1])/(Target.TemperatureArray[TIndex]-Target.TemperatureArray[TIndex-1])
                    if CurrentP>-5:
                        co_p = (CurrentP-Target.PressureArray[PIndex-1])/(Target.PressureArray[PIndex]-Target.PressureArray[PIndex-1])
                    else:
                        co_p = 0.0

                    assert -1e-16<co_t<1.000000001
                    assert -1e-16<co_p<1.000000001



                    if co_p>0:
                        FirstTerm = Target.CrossSectionData[TIndex-1, PIndex-1,:,:]@Target.nz[:, self.CurrentLayer]
                        SecondTerm = Target.CrossSectionData[TIndex-1, PIndex,:,:]@Target.nz[:, self.CurrentLayer]
                        ThirdTerm = Target.CrossSectionData[TIndex, PIndex-1,:,:]@Target.nz[:, self.CurrentLayer]
                        FourthTerm = Target.CrossSectionData[TIndex, PIndex,:,:]@Target.nz[:, self.CurrentLayer]

                        self.alpha[:,self.CurrentLayer] = ((1-co_t)*(1-co_p))*FirstTerm + \
                                                        ((1-co_t)*co_p)*SecondTerm +  \
                                                        (co_t*(1-co_p))*ThirdTerm + \
                                                        (co_t*co_p)*FourthTerm


                    elif co_p == 0:
                        FirstTerm = Target.CrossSectionData[TIndex-1, PIndex,:,:]@Target.nz[:, self.CurrentLayer]
                        ThirdTerm = Target.CrossSectionData[TIndex, PIndex,:,:]@Target.nz[:, self.CurrentLayer]

                        self.alpha[:,self.CurrentLayer] = (1-co_t)*FirstTerm + \
                                                        co_t*ThirdTerm

                    if self.CIA:
                        #print("the value of co_t is::", co_t)
                        self.alpha[:,self.CurrentLayer] +=   \
                        (Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[TIndex,0,:]+(co_t)*Target.CIA_CS[TIndex+1,0,:]) + \
                        (Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer]*9./90.)*((1-co_t)*Target.CIA_CS[TIndex,1,:]+(co_t)*Target.CIA_CS[TIndex+1,1,:]) + \
                        (Target.nz_N2_ama[self.CurrentLayer]*Target.nz_N2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[TIndex,2,:]+(co_t)*Target.CIA_CS[TIndex+1,2,:])

                        SelectIndex = self.alpha[:, self.CurrentLayer]<0
                        self.alpha[SelectIndex, self.CurrentLayer] = 0.0




                elif "hill" in interpolation.lower():
                    if PIndex>0.5:
                        Temp1, Temp2 = [Target.TemperatureArray[TIndex-1], Target.TemperatureArray[TIndex]]
                        P1, P2 = [Target.PressureArray[PIndex-1], Target.PressureArray[PIndex]]

                        #See if they are exactly same
                        m = (CurrentP-P1)/(P2-P1)

                        assert -1e-16<m<1.000000001
                        Sigma11 = Target.CrossSectionData[TIndex-1, PIndex-1, :]
                        Sigma12 = Target.CrossSectionData[TIndex-1, PIndex, :]
                        Sigma21 = Target.CrossSectionData[TIndex, PIndex-1, :]
                        Sigma22 = Target.CrossSectionData[TIndex, PIndex, :]

                    elif PIndex == 0:
                        #See if they are exactly same
                        m = 0.0
                        Sigma11 = Target.CrossSectionData[TIndex-1, PIndex, :]
                        Sigma12 = Target.CrossSectionData[TIndex-1, PIndex, :]
                        Sigma21 = Target.CrossSectionData[TIndex, PIndex, :]
                        Sigma22 = Target.CrossSectionData[TIndex, PIndex, :]

                    #Performing hill interpolation
                    UndSigma1 = Sigma11+ m*(Sigma12-Sigma11)
                    UndSigma2 = Sigma21+ m*(Sigma22-Sigma21)

                    RatioSigma = UndSigma1/UndSigma2
                    bi = 1./(1./Temp2-1./Temp1)*np.log(RatioSigma)
                    ai = UndSigma1*np.exp(bi/Temp1)

                    self.CurrentInterpSigma = ai*np.exp(-bi/CurrentT)


                    ZeroIndex = np.logical_or(np.isnan(self.CurrentInterpSigma), \
                                            ~np.isfinite(self.CurrentInterpSigma))

                    #Replace nan with zeros
                    self.CurrentInterpSigma[ZeroIndex] = 0.0
                    self.alpha[:,self.CurrentLayer] = np.matmul(self.CurrentInterpSigma, Target.nz[:, self.CurrentLayer])


                    if self.CIA:
                        self.alpha[:,self.CurrentLayer] +=   \
                                (Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[TIndex,0,:]+(co_t)*Target.CIA_CS[TIndex+1,0,:]) + \
                                (Target.nz_H2_ama[self.CurrentLayer]*Target.nz_H2_ama[self.CurrentLayer]*9./90.)*((1-co_t)*Target.CIA_CS[TIndex,1,:]+(co_t)*Target.CIA_CS[TIndex+1,1,:]) + \
                                (Target.nz_N2_ama[self.CurrentLayer]*Target.nz_N2_ama[self.CurrentLayer])*((1-co_t)*Target.CIA_CS[TIndex,2,:]+(co_t)*Target.CIA_CS[TIndex+1,2,:])
                        SelectIndex = self.alpha[:, self.CurrentLayer]<0
                        self.alpha[SelectIndex, self.CurrentLayer] = 0.0
                            
                else:
                    raise ValueError("Use either bilinear/hill interpolation.")




            sz = Target.NumLayers


            self.Spectrum = ((Target.Rp)**2+ \
                            2.0*np.matmul(1.0-(np.exp(-(2.0*(np.matmul(self.alpha[:,0:sz-1],np.transpose(self.ds_[:,:sz-1])))))), \
                            (Target.Rp+Target.zValuesCm[:sz-1])*np.transpose(self.dz_cm[:sz-1])))/Target.Rs**2
            self.Spectrum = self.Spectrum.flatten()

            ##Following two are equivaluent
            self.SpectrumHeight = 0.5*(self.Spectrum*Target.Rs**2/Target.Rp-Target.Rp)

            #First is where all the layers are included    
            self.AllSpectrum[:,Outerlayer+1] = self.SpectrumHeight

        print("Now calculate the contribution function")    
        
        self.ContributionFunction = np.zeros(Target.NumLayers, dtype=np.float64)

        R_Nom = self.AllSpectrum[:,0]

        for i in range(Target.NumLayers):
            R = self.AllSpectrum[:,i+1]
            self.ContributionFunction[i] = np.sum(R_Nom**2-R**2)

        TotalSum = np.sum(self.ContributionFunction)
        self.ContributionFunction = self.ContributionFunction/TotalSum

        LogPressure = np.log10(Target.PzAnalytical)    

        Area = np.trapz(LogPressure, self.ContributionFunction)
        print(Area)

        #Normalize the area to 1 so the total contribution function is 100%

        plt.figure()
        plt.plot(Target.PzAnalytical, self.ContributionFunction/Area, "k-")
        plt.xscale('log')
        plt.show()

        return self.ContributionFunction


            
