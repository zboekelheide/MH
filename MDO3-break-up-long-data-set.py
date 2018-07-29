import os
import sys
import numpy
import math
import scipy
#from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
#import time
#import visa
from time import gmtime, strftime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})


#frequency = 225000

basedir=r'''C:\Users\boekelhz\a1Lafayette\data\TektronixMDO3024\pythonMH\20180705'''
subf = '150020'
os.chdir(basedir)

numfiles = 42
addnum = 4
for i in range(numfiles):
    bigfilenm = 'synomag-w-FO-decreasing-'+str(i+addnum)+'.txt'


    #datesubf = strftime("%Y%m%d")
    #timesubf = strftime("%H%M%S")
    #kstep=0
    outdir = basedir+'\\'''+subf+'\\dataselections\\'''
    if not os.path.exists(outdir):
        print('Creating directory')
        os.makedirs(outdir)  
    
    
    def flatten(l):
      out = []
      for item in l:
        if isinstance(item, (list, tuple)):
          out.extend(flatten(item))
        else:
          out.append(item)
      return out
    
    ambrelldata = pd.read_csv(subf+'\\'+bigfilenm)
    
    
    times = flatten(ambrelldata.iloc[:,0:1].values.tolist())
    #    print('times float: '+str(times[0:10]))
    M = flatten(ambrelldata.iloc[:,1:2].values.tolist())
    #    print('M: '+str(M[0:10]))
    H = flatten(ambrelldata.iloc[:,2:3].values.tolist())
    #for i in range(len(M)):
    #    M[i] = polarity*M[i]
    
    
    
    total_points = len(M)
    timestep = (times[total_points - 1]-times[0])/(total_points-1)
    print('timestep: '+str(timestep))
    
    
    
    bigHspectrum = numpy.fft.fft(H)
    #bigHspectrum = numpy.fft.fft(H)
    bigfreq = numpy.fft.fftfreq(total_points, d=timestep) 
    
    fundindex = numpy.argmax(numpy.abs(bigHspectrum[1:(int(total_points/2))]))+1
    guess_freq = bigfreq[fundindex]
    
    
    
    period = 1/guess_freq
    tsteps_in_period = int(period/timestep)
    print('tsteps_in_period: '+str(tsteps_in_period))
    #    tsteps_in_period = 11161    
    
    
    plt.figure(3)
    plt.style.use('classic')
    plt.title('H')
    plt.grid(True)
    plt.xlabel('data points')
    plt.ylabel(r'H')
    plt.xticks(rotation=90)
    plt.plot(range(total_points), H, 'b.', label = r'M')
    #plt.ylim(0, 2*pi)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    
    #This section chooses the number of periods to analyze (an integer). It can be
    #implemented as an interactive session with the user, or the number of
    #periods can be entered as an input parameter (how it is currently set up).    
    print('Here is the Ambrell data. Choose a region you would like to analyze. The number of data poins per period is approximately '+str(tsteps_in_period))
    
    selectionproceed = 'n'
    k = 0
    new_upper = 0
    while selectionproceed != 'q':
    
        lower = input("Enter the data point numbers for the beginning of the reigon. Note last region ended at "+str(new_upper)+'.')
        lower = int(lower)
        upper= input("Enter the data point numbers for the end of the reigon.")
        upper = int(upper)
        if upper < lower:
            print('upper bound must be greater than lower bound.')
            continue
        
    #        lower = 0
    #        upper = 999999
        
        est_num_periods = int(numpy.floor((upper - lower)/tsteps_in_period))
    
    #    lower = 0
    #    upper = est_num_periods*tsteps_in_period-1
    
    #    adj_total_points = est_num_periods*tsteps_in_period #+ extra_points
    #    new_upper = lower+adj_total_points
        new_upper = upper
        #(worry about integer number of periods later)
    
        '''
        plt.figure(3+k*4)
        plt.style.use('classic')
        plt.title('M')
        plt.grid(True)
        plt.xlabel('data points')
        plt.ylabel(r'M')
        plt.xticks(rotation=90)
        plt.plot(range(total_points), M, 'b.', label = r'M')
        plt.plot(range(lower, new_upper), M[lower:new_upper], 'go', label = descriptor+' M selection')
        #plt.ylim(0, 2*pi)
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()
        '''
        plt.figure(4+k*4)
        plt.style.use('classic')
        plt.title('M')
        plt.grid(True)
        plt.xlabel('data points')
        plt.ylabel(r'M')
        plt.xticks(rotation=90)
        plt.plot(range(lower, new_upper), M[lower:new_upper], 'g.', label = ' M selection')
        plt.plot(range(lower, new_upper), H[lower:new_upper], 'b.', label = ' H selection')
        #plt.ylim(0, 2*pi)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()
        '''
        plt.figure(5+k*4)
        plt.style.use('classic')
        plt.title('first period M')
        plt.grid(True)
        plt.xlabel('data points')
        plt.ylabel(r'M')
        plt.xticks(rotation=90)
        datapoints = numpy.arange(lower, first_per_upper, 1).tolist()
        plt.plot(datapoints, M[lower:first_per_upper], 'g.', label = descriptor+' first period M selection')
        plt.plot(datapoints, H[lower:first_per_upper], 'b.', label = descriptor+' first period H selection')
        #plt.ylim(0, 2*pi)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()
    
        plt.figure(6+k*4)
        plt.style.use('classic')
        plt.title('last period M')
        plt.grid(True)
        plt.xlabel('data points')
        plt.ylabel(r'M')
        plt.xticks(rotation=90)
        datapoints = numpy.arange(last_per_lower, upper, 1).tolist()
        plt.plot(datapoints, M[last_per_lower:upper], 'g.', label = descriptor+' last period M selection')
        plt.plot(datapoints, H[last_per_lower:upper], 'b.', label = descriptor+' last period H selection')
        #plt.ylim(0, 2*pi)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()
        '''    
        selectionproceed = input('Are you ready to proceed with this selection of data? y to proceed, n to enter a different selection, q to quit.')
        #selectionproceed = 'y'
        if selectionproceed == 'n':
           continue
        elif selectionproceed == 'y':
            filenm = bigfilenm+str(k)
            outfile = outdir+'\\'+filenm
            writefile=open(outfile, 'a')
            writefile.write('Time (s), Ch1 (V), Ch2 (V) \n')
            for i in range(new_upper-lower):
                writefile.write(str(times[lower+i])+', '+str(M[lower+i])+', '+str(H[lower+i])+'\n')
            k+=1

