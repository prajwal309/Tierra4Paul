import matplotlib.pyplot as plt
import numpy as np
import os
from tierra import Target
from tierra.transmission import TransmissionSpectroscopy


#Load the custom PT profile in order to address 
PT_Data = np.loadtxt("TitanData/HASI_L4_ATMO_PROFILE_COMPLETE.TAB", delimiter=";")

Time = PT_Data[:,0]               #Time when it was taken  
Height = PT_Data[:,1]             #The height of the proble  
Pressure = PT_Data[:,2]           #Log of the pressure
Temperature = PT_Data[:,3]        #The temperature with the height
Density = PT_Data[:,4]            #The density with the height  


Coeffs_3 = np.polyfit(np.log10(Pressure), Temperature, 3)
ModelFit_3 = np.polyval(Coeffs_3, np.log10(Pressure))

Coeffs_4 = np.polyfit(np.log10(Pressure), Temperature, 4)
ModelFit_4 = np.polyval(Coeffs_4, np.log10(Pressure))

Coeffs_5 = np.polyfit(np.log10(Pressure), Temperature, 5)
ModelFit_5 = np.polyval(Coeffs_5, np.log10(Pressure))

Coeffs_6 = np.polyfit(np.log10(Pressure), Temperature, 6)
ModelFit_6 = np.polyval(Coeffs_6, np.log10(Pressure))

Coeffs_7 = np.polyfit(np.log10(Pressure), Temperature, 7)
ModelFit_7 = np.polyval(Coeffs_7, np.log10(Pressure))


#Sho
plt.figure(figsize=(12,8))
plt.plot(Temperature, np.log10(Pressure), "ko")
plt.plot(ModelFit_3, np.log10(Pressure), "r-", label="Order-3")
plt.plot(ModelFit_4, np.log10(Pressure), "b-", label="Order-4")
plt.plot(ModelFit_5, np.log10(Pressure), "g-", label="Order-5")
plt.plot(ModelFit_6, np.log10(Pressure), "y-", label="Order-6")
plt.plot(ModelFit_7, np.log10(Pressure), color="tomato", marker="None", linestyle=":", label="Order-7")
plt.legend()
plt.ylabel("Pressure [Pa]", fontsize=20)
plt.xlabel("Temperature [K]", fontsize=20)
plt.gca().invert_yaxis()
plt.savefig("Figures/Titan_PT_Profile.png")
plt.close()

input("")
Planet1 = Target.System(LoadFromFile=True)


#Now use custom profile
Planet1.PT_Profile(zStep=0.1, type="custom", ShowPlot=False)

input("We will wait here...")

print("The mean molecular mass of the atmosphere is:", Planet1.mu)


CurrentDirectory = os.getcwd()
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
