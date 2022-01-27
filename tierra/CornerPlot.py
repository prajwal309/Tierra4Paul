import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

import matplotlib
matplotlib.use('Tkagg')



import matplotlib as mpl
mpl.rc('font',**{'family':'sans-serif', 'serif':['Computer Modern Serif'],'sans-serif':['Helvetica'], 'size':15,'weight':'bold'})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':2.0})
mpl.rc('ytick',**{'major.pad':22, 'color':'k'})
mpl.rc('xtick',**{'major.pad':10,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text', **{'usetex':True})
mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
mpl.rc('contour', **{'negative_linestyle':'solid'})


def CustomPlot(x , y, ax = None, sort = True, bins=50, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    fig,ax = plt.subplots(figsize=(12,10))
    ax.scatter(x, y, c=z, s=2)

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    #cbar.ax.set_ylabel('Density')

    return ax





def FormatLabel(LabelsX):
    LabelsX = np.array(LabelsX)
    Diff=np.mean(np.diff(LabelsX))
    if min(abs(LabelsX))<10.0 and min(abs(LabelsX))>0.001:
        LabelStr = ["%.2f" %(Value) for Value in LabelsX]
    elif min(abs(LabelsX))<1000 and min(abs(LabelsX))>1:
        LabelStr = ["%.0d" %(Value) for Value in LabelsX]
    else:
        LabelStr = ["%.2e" %(Value) for Value in LabelsX]
    return LabelStr


def CustomCornerPlot(Data, Parameters, Values=None):

    NDim = len(Data)
    FigDim = NDim*2.5

    if len(Parameters)>0:
        try:
          assert len(Parameters) == NDim
        except:
          raise("The number of should match the dimension of the data provided.")


    fig, ax = plt.subplots(NDim, NDim, figsize=(FigDim, FigDim), dpi=80)


    for i in range(NDim):
        for j in range(NDim):
            if j<i:

                NBins = 10
                counts,xbins,ybins=np.histogram2d(Data[i,:], Data[j,:],bins=NBins)
                Levels = np.percentile(counts,[68.27,95.45,99.73])

                #labels = np.round(np.linspace(min(xbins), max(xbins),6),2)

                #good options for colormap are gist_earth_r, gray_r
                ax[i,j].hist2d(Data[i,:], Data[j,:], orientation='horizontal', cmap='Reds', bins = 2*NBins)#, norm=PowerNorm(gamma=0.5))
                ax[i,j].contour(counts.transpose(),Levels,extent=[xbins.min(),xbins.max(),
                    ybins.min(),ybins.max()],linewidths=2,cmap="Reds",
                    linestyles='-')

                if Values:
                    ax[i,j].plot(Values[i],Values[j], "r+", markersize=30,  markeredgewidth=3)


                #Format the labels
                NumLabels = 5
                StepSizeX = (max(xbins) - min(xbins))/NumLabels
                StepSizeY = (max(ybins) - min(ybins))/NumLabels

                LabelsX = np.linspace(min(xbins)+StepSizeX, max(xbins)-StepSizeX, NumLabels)
                LabelsXStr = FormatLabel(LabelsX)

                LabelsY = np.linspace(min(ybins)+StepSizeY, max(ybins)-StepSizeY, NumLabels)
                LabelsYStr = FormatLabel(LabelsX)

                ax[i,j].set_xticks(LabelsX)
                ax[i,j].set_xticklabels(LabelsXStr, rotation=45)
                ax[i,j].set_xlim(min(xbins), max(xbins))

                ax[i,j].set_yticks(LabelsY)
                ax[i,j].set_yticklabels(LabelsYStr, rotation=45)
                ax[i,j].set_ylim(min(ybins), max(ybins))
                ax[i,j].tick_params(which="both", direction="in", pad=5)
            elif i==j:
                print("The value of i is::", i)
                print("The value of j is::", j)
                print("The shape of data is given by::", np.shape(Data))
                print("The length of the data is given by::", len(Data[i, :]))
                ax[i,j].hist(Data[i,:], fill=False, histtype='step', linewidth=2, color="navy", normed=True)
                PercentileValues = np.percentile(Data[i,:],[15.8, 50.0, 84.2])
                for counter_pc, Value in enumerate(PercentileValues):
                    if counter_pc == 1:
                            ax[i,j].axvline(x=Value, color="red",  lw=1.5)
                    else:
                            ax[i,j].axvline(x=Value, color="cyan",  linestyle="--", lw=2.5)

                #assign the title
                Median = PercentileValues[1]

                if Median<100 and Median>0.001:
                    MedianStr = "%0.2f" %Median
                else:
                    MedianStr = "%0.2e" %Median

                UpperError = PercentileValues[2] - PercentileValues[1]
                if UpperError<100 and UpperError>0.001:
                    UpperErrorStr = "%0.2f" %UpperError
                else:
                    UpperErrorStr = "%0.2e" %UpperError

                LowerError = PercentileValues[1] - PercentileValues[0]
                if LowerError<100 and LowerError>0.001:
                    LowerErrorStr = "%0.2f" %LowerError
                else:
                    LowerErrorStr = "%0.2e" %LowerError

                Title = Parameters[i]+ " = %s$^{+%s}_{-%s}$" %(MedianStr, UpperErrorStr, LowerErrorStr)
                print(Title)
                ax[i,j].set_title(Title)
                ax[i,j].tick_params(which="both", direction="in", pad=5)
            else:
                ax[i,j].set_visible(False)
                ax[i,j].tick_params(which="both", direction="in", pad=5)


            #Now for the ylabels
            if j!=0 or i==j:
                ax[i,j].set_yticklabels([])


            #Now for the xlabels
            if i!=NDim-1 or i==j:
                ax[i,j].set_xticklabels([])


            #assign the title

            #

    plt.subplots_adjust(wspace=0.025, hspace=0.025, left = 0.05,
    right = 0.95, bottom = 0.05, top = 0.95)
    plt.savefig("Trial.png")
    plt.savefig("Trial.pdf", format='pdf')
    plt.show()


def DoubleCustomCornerPlot(Data1, Data2, Parameters, Values=None, CMapList=None, colorList=None, SaveName=None):

    if not(CMapList):
        CMap = ["Reds", "Blues"]
        colorList = ["red", "blue"]
    else:
        CMap = CMapList
        colorList = colorList

    NDim = len(Data1)
    assert len(Data2) == NDim
    FigDim = NDim*2.5

    if len(Parameters)>0:
        try:
          assert len(Parameters) == NDim
        except:
          raise("The number of should match the dimension of the data provided.")


    fig, ax = plt.subplots(NDim, NDim, figsize=(FigDim, FigDim), dpi=80)


    for i in range(NDim):
        for j in range(NDim):
            if j<i:

                NBins = 10

                #counts1,xbins1,ybins1=np.histogram2d(Data1[i,:], Data1[j,:],bins=NBins)
                #counts2,xbins2,ybins2=np.histogram2d(Data2[i,:], Data2[j,:],bins=NBins)

                x1 = Data1[i,:]
                y1 = Data1[j,:]
                x2 = Data2[i,:]
                y2 = Data2[j,:]

                #For the first set of data
                data1, x_e1, y_e1 = np.histogram2d(x1, y1, bins=NBins, density=True )
                z1 = interpn( ( 0.5*(x_e1[1:] + x_e1[:-1]) , 0.5*(y_e1[1:]+y_e1[:-1]) ) , data1, np.vstack([x1,y1]).T , method = "splinef2d", bounds_error = False)

                #To be sure to plot all data
                z1[np.where(np.isnan(z1))] = 0.0

                # Sort the points by density, so that the densest points are plotted last
                idx1 = z1.argsort()
                x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]

                #For the second set of data
                data2, x_e2, y_e2 = np.histogram2d(x2, y2, bins=NBins, density=True )
                z2 = interpn( ( 0.5*(x_e2[1:] + x_e2[:-1]) , 0.5*(y_e2[1:]+y_e2[:-1]) ) , data2, np.vstack([x2,y2]).T , method = "splinef2d", bounds_error = False)

                #To be sure to plot all data
                z2[np.where(np.isnan(z2))] = 0.0

                # Sort the points by density, so that the densest points are plotted last
                idx2 = z2.argsort()
                x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]

                #Levels1 = np.percentile(counts1,[68.27,95.45,99.73])
                #Levels2 = np.percentile(counts2,[68.27,95.45,99.73])

                #good options for colormap are gist_earth_r, gray_r

                #norm = Normalize(vmin = np.min(z), vmax = np.max(z))

                ax[i,j].scatter(y1, x1, c=z1, s=2, cmap=CMap[0], alpha=0.1)
                ax[i,j].scatter(y2, x2, c=z2, s=2, cmap=CMap[1], alpha=0.1)


                #ax[i,j].contourf((y1, x1), z1)
                #ax[i,j].contourf((y2, x2), z2)


                if Values:
                    ax[i,j].plot(Values[j], Values[i], "k+", lw=2, markersize=50,  markeredgewidth=3)



            elif i==j:
                ax[i,j].hist(Data1[i,:], fill=False, histtype='step', linewidth=2, color=colorList[0], normed=True)
                ax[i,j].hist(Data2[i,:], fill=False, histtype='step', linewidth=2, color=colorList[1], normed=True)

                PercentileValues1 = np.percentile(Data1[i,:],[15.8, 50.0, 84.2])
                PercentileValues2 = np.percentile(Data2[i,:],[15.8, 50.0, 84.2])

                for counter_pc in range(len(PercentileValues1)):

                    Value1 = PercentileValues1[counter_pc]
                    Value2 = PercentileValues2[counter_pc]

                    if counter_pc == 1:
                        ax[i,j].axvline(x=Value1, color=colorList[0],  linestyle=":", lw=1.5, alpha=0.90)
                        ax[i,j].axvline(x=Value2, color=colorList[1], linestyle=":", lw=1.5, alpha=0.90)
                        if Values:
                            ax[i,j].axvline(Values[i],Values[j], color="black", lw=4, markersize=100)
                    else:
                        ax[i,j].axvline(x=Value1, color=colorList[0],  linestyle="--", alpha=0.5, lw=2.5)
                        ax[i,j].axvline(x=Value2, color=colorList[1],  linestyle="--", alpha=0.5, lw=2.5)

                #assign the title
                Median = PercentileValues1[1]

                if np.abs(Median)<500 and np.abs(Median)>0.001:
                    MedianStr = "%0.2f" %Median
                else:
                    MedianStr = "%0.2e" %Median

                UpperError = PercentileValues1[2] - PercentileValues1[1]
                if np.abs(UpperError)<100 and np.abs(UpperError)>0.001:
                    UpperErrorStr = "%0.2f" %UpperError
                else:
                    UpperErrorStr = "%0.2e" %UpperError

                LowerError = PercentileValues1[1] - PercentileValues1[0]
                if np.abs(LowerError)<100 and np.abs(LowerError)>0.001:
                    LowerErrorStr = "%0.2f" %LowerError
                else:
                    LowerErrorStr = "%0.2e" %LowerError

                Title = Parameters[i]+ " = %s$^{+%s}_{-%s}$" %(MedianStr, UpperErrorStr, LowerErrorStr)
                print(Title)
                ax[i,j].set_title(Title, loc='left', color=colorList[0])#, pad=130)#, rotation=30)

            else:
                ax[i,j].set_visible(False)
                #ax[i,j].axis('equal')
                ax[i,j].set_aspect('equal', 'box')
            #Now for the ylabels
            if j!=0 or i==j:
                ax[i,j].set_yticklabels([])


            #Now for the xlabels
            if i!=NDim-1 or i==j:
                ax[i,j].set_xticklabels([])

            ax[i,j].tick_params(which="both", direction="in")

            #assign the title

            #
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.035, hspace=0.035, left = 0.08,
    right = 0.95, bottom = 0.08, top = 0.89)

    if not(SaveName):
        plt.savefig("Figures/CornerPlot.png")
        #plt.savefig("CornerPlot.pdf", format='pdf')
    else:
        plt.savefig("Figures/"+SaveName+"_Corner.png")
        plt.savefig("Figures/"+SaveName+"_Corner.pdf")
        #plt.savefig("CornerPlot.pdf", format='pdf')
    plt.close('all')





def SingleCustomCornerPlot(Data1, Parameters, Values=None, CMapList=None, colorList=None, SaveName=None):

    if not(CMapList):
        CMap = ["Reds", "Blues"]
        colorList = ["red", "blue"]
    else:
        CMap = CMapList
        colorList = colorList

    NDim = len(Data1)
    FigDim = NDim*2.5

    if len(Parameters)>0:
        try:
          assert len(Parameters) == NDim
        except:
          raise("The number of should match the dimension of the data provided.")


    fig, ax = plt.subplots(NDim, NDim, figsize=(FigDim, FigDim), dpi=80)


    for i in range(NDim):
        for j in range(NDim):
            if j<i:

                NBins = 10

                x1 = Data1[i,:]
                y1 = Data1[j,:]


                #For the first set of data
                data1, x_e1, y_e1 = np.histogram2d(x1, y1, bins=NBins, density=True )
                z1 = interpn( ( 0.5*(x_e1[1:] + x_e1[:-1]) , 0.5*(y_e1[1:]+y_e1[:-1]) ) , data1, np.vstack([x1,y1]).T , method = "splinef2d", bounds_error = False)

                #To be sure to plot all data
                z1[np.where(np.isnan(z1))] = 0.0

                # Sort the points by density, so that the densest points are plotted last
                idx1 = z1.argsort()
                x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]

                ax[i,j].scatter(y1, x1, c=z1, s=2, cmap=CMap[0], alpha=0.1)


                if Values:
                    ax[i,j].plot(Values[j], Values[i], "k+", lw=2, markersize=50,  markeredgewidth=3)



            elif i==j:
                ax[i,j].hist(Data1[i,:], fill=False, histtype='step', linewidth=2, color=colorList[0], normed=True)

                PercentileValues1 = np.percentile(Data1[i,:],[15.8, 50.0, 84.2])

                for counter_pc in range(len(PercentileValues1)):

                    Value1 = PercentileValues1[counter_pc]


                    if counter_pc == 1:
                        ax[i,j].axvline(x=Value1, color=colorList[0],  linestyle=":", lw=1.5, alpha=0.90)
                        if Values:
                            ax[i,j].axvline(Values[i],Values[j], color="black", lw=4, markersize=100)
                    else:
                        ax[i,j].axvline(x=Value1, color=colorList[0],  linestyle="--", alpha=0.5, lw=2.5)


                #assign the title
                Median = PercentileValues1[1]

                if np.abs(Median)<500 and np.abs(Median)>0.001:
                    MedianStr = "%0.2f" %Median
                else:
                    MedianStr = "%0.2e" %Median

                UpperError = PercentileValues1[2] - PercentileValues1[1]
                if np.abs(UpperError)<100 and np.abs(UpperError)>0.001:
                    UpperErrorStr = "%0.2f" %UpperError
                else:
                    UpperErrorStr = "%0.2e" %UpperError

                LowerError = PercentileValues1[1] - PercentileValues1[0]
                if np.abs(LowerError)<100 and np.abs(LowerError)>0.001:
                    LowerErrorStr = "%0.2f" %LowerError
                else:
                    LowerErrorStr = "%0.2e" %LowerError

                Title = Parameters[i]+ " = %s$^{+%s}_{-%s}$" %(MedianStr, UpperErrorStr, LowerErrorStr)
                print(Title)
                ax[i,j].set_title(Title, loc='left', color=colorList[0])#, pad=130)#, rotation=30)

            else:
                ax[i,j].set_visible(False)
                #ax[i,j].axis('equal')
                ax[i,j].set_aspect('equal', 'box')
            #Now for the ylabels
            if j!=0 or i==j:
                ax[i,j].set_yticklabels([])


            #Now for the xlabels
            if i!=NDim-1 or i==j:
                ax[i,j].set_xticklabels([])

            ax[i,j].tick_params(which="both", direction="in")

            #assign the title

            #
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.035, hspace=0.035, left = 0.08,
    right = 0.95, bottom = 0.08, top = 0.89)

    if not(SaveName):
        plt.savefig("Figures/CornerPlot.png")
        #plt.savefig("CornerPlot.pdf", format='pdf')
    else:
        plt.savefig("Figures/"+SaveName+"_Corner.png")
        plt.savefig("Figures/"+SaveName+"_Corner.pdf")
    plt.close('all')








def AllContourPlot(Data1, Data2, Data3, Data4, Data5, Data6, Data7, Parameters, Values=None, CMapList=None, colorList=None, SaveName=None):


    NDim = len(Data1)
    FigDim = NDim*2.5

    if len(Parameters)>0:
        try:
          assert len(Parameters) == NDim
        except:
          raise("The number of should match the dimension of the data provided.")


    fig, ax = plt.subplots(NDim, NDim, figsize=(FigDim, FigDim), dpi=80)


    for i in range(NDim):
        for j in range(NDim):
            if j<i:

                x1 = Data1[i,:]
                y1 = Data1[j,:]

                x2 = Data2[i,:]
                y2 = Data2[j,:]

                x3 = Data3[i,:]
                y3 = Data3[j,:]

                x4 = Data4[i,:]
                y4 = Data4[j,:]

                x5 = Data5[i,:]
                y5 = Data5[j,:]

                x6 = Data6[i,:]
                y6 = Data6[j,:]

                x7 = Data7[i,:]
                y7 = Data7[j,:]

                #ax[i,j].scatter(y1,x1)

                sns.kdeplot(y1, x1, cmap=CMapList[0], ax=ax[i,j], thresh=0.05, levels=2)
                sns.kdeplot(y2, x2, cmap=CMapList[1], ax=ax[i,j], thresh=0, levels=3)
                sns.kdeplot(y3, x3, cmap=CMapList[2], ax=ax[i,j], thresh=0, levels=3)
                sns.kdeplot(y4, x4, cmap=CMapList[3], ax=ax[i,j], thresh=0, levels=3)
                sns.kdeplot(y5, x5, cmap=CMapList[4], ax=ax[i,j], thresh=0, levels=3)
                sns.kdeplot(y6, x6, cmap=CMapList[5], ax=ax[i,j], thresh=0, levels=3)
                sns.kdeplot(y7, x7, cmap=CMapList[6], ax=ax[i,j], thresh=0, levels=3)

                if Values:
                    ax[i,j].plot(Values[j], Values[i], "k+", lw=2, markersize=50,  markeredgewidth=3)



            elif i==j:
                ax[i,j].hist(Data1[i,:], fill=False, histtype='step', linewidth=2, color=colorList[0], normed=True)
                ax[i,j].hist(Data2[i,:], fill=False, histtype='step', linewidth=2, color=colorList[1], normed=True)
                ax[i,j].hist(Data3[i,:], fill=False, histtype='step', linewidth=2, color=colorList[2], normed=True)
                ax[i,j].hist(Data4[i,:], fill=False, histtype='step', linewidth=2, color=colorList[3], normed=True)
                ax[i,j].hist(Data5[i,:], fill=False, histtype='step', linewidth=2, color=colorList[4], normed=True)
                ax[i,j].hist(Data6[i,:], fill=False, histtype='step', linewidth=2, color=colorList[5], normed=True)
                ax[i,j].hist(Data7[i,:], fill=False, histtype='step', linewidth=2, color=colorList[6], normed=True)

                PercentileValues1 = np.percentile(Data1[i,:],[15.8, 50.0, 84.2])
                ax[i,j].axvline(Values[i],Values[j], color="black", lw=4, zorder=100)



                #assign the title
                Median = PercentileValues1[1]

                if np.abs(Median)<500 and np.abs(Median)>0.001:
                    MedianStr = "%0.2f" %Median
                else:
                    MedianStr = "%0.2e" %Median

                UpperError = PercentileValues1[2] - PercentileValues1[1]
                if np.abs(UpperError)<100 and np.abs(UpperError)>0.001:
                    UpperErrorStr = "%0.2f" %UpperError
                else:
                    UpperErrorStr = "%0.2e" %UpperError

                LowerError = PercentileValues1[1] - PercentileValues1[0]
                if np.abs(LowerError)<100 and np.abs(LowerError)>0.001:
                    LowerErrorStr = "%0.2f" %LowerError
                else:
                    LowerErrorStr = "%0.2e" %LowerError

                Title = Parameters[i]+ " = %s$^{+%s}_{-%s}$" %(MedianStr, UpperErrorStr, LowerErrorStr)
                print(Title)
                ax[i,j].set_title(Title, loc='left', color=colorList[0])#, pad=130)#, rotation=30)

            else:
                ax[i,j].set_visible(False)
                #ax[i,j].axis('equal')
                ax[i,j].set_aspect('equal', 'box')
            #Now for the ylabels
            if j!=0 or i==j:
                ax[i,j].set_yticklabels([])


            #Now for the xlabels
            if i!=NDim-1 or i==j:
                ax[i,j].set_xticklabels([])

            ax[i,j].tick_params(which="both", direction="in")

            #assign the title

            #
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.035, hspace=0.035, left = 0.08,
    right = 0.95, bottom = 0.08, top = 0.89)

    if not(SaveName):
        plt.savefig("Figures/CornerPlot.png")
        #plt.savefig("CornerPlot.pdf", format='pdf')
    else:
        plt.savefig("Figures/"+SaveName+"_Corner.png")
        plt.savefig("Figures/"+SaveName+"_Corner.pdf")
    plt.close('all')
