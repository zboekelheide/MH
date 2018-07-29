import os
#import sys
import numpy
import math
import scipy
#from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import time
#import visa
from time import strftime
import pandas as pd
import matplotlib.pyplot as plt
#rcParams.update({'figure.autolayout': True})

#Stuff for doing fft analysis etc
pi = math.pi 
inf = numpy.inf

polarity = -1

M_calib_factor = -9.7e6
H_calib_factor = -6.4e7
# in units of kA/m per V-s

def odd_harmonic_M(length, i, est_num_periods):
    oddnums = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]
    if (i % est_num_periods == 0) and ((round(i/est_num_periods) in oddnums and i < length/2) or (round((length - i)/est_num_periods) in oddnums and i > length/2)):
            return(1)
    else:
        return(0)

def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c

#def sinseries(t, A1, A2, A3, A4, w, p, c):  return A1 * numpy.sin(w*t + p) + A2 * numpy.sin(2*w*t + p) + A3 * numpy.sin(3*w*t + p) + A4 * numpy.sin(4*w*t + p) + c
    
#def fit_sin_series(tt, yy, w):
#    tt = numpy.array(tt)
#    yy = numpy.array(yy)
#    guess_amp = numpy.std(yy) * 2.**0.5
#    guess_offset = numpy.mean(yy)
#    guess = numpy.array([guess_amp, guess_amp/4, guess_amp/4, guess_amp/4, w, 0., guess_offset])
#    param_bounds =([-inf,-inf,-inf,-inf,w-1,-inf,-inf],[inf,inf,inf,inf,w+1,inf,inf])
#    popt, pcov = scipy.optimize.curve_fit(sinseries, tt, yy, p0=guess, bounds=param_bounds)
#    A1, A2, A3, A4, w, p, c = popt
#    return [A1, A2, A3, A4, w, p, c]

def fit_sin(tt, yy):
    #Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset"
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(numpy.abs(Fyy[1:]))+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = 1.41*numpy.std(yy)
    guess_w = 2.*numpy.pi*guess_freq
    guess_offset = numpy.mean(yy)
    ind1 = int(len(yy)/4)
    ind2 = int(len(yy)/8)
    arccosarg = (yy[ind1] - yy[0])/((tt[ind1] - tt[0])*guess_amp*guess_w)
    if arccosarg > 1:
        arccosarg = 1
    if arccosarg < -1:
        arccosarg = -1
    guess_phase = numpy.arccos(arccosarg)-pi_mod(guess_w*tt[ind2])
    guess = numpy.array([guess_amp, guess_w, guess_phase, guess_offset])
    #def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev = 200000)
    except:
        popt = [guess_amp, guess_w, guess_phase, guess_offset]
    A, w, p, c = popt
    return [A, w, p, c]

def pi_mod_array(phasearray):
    for k in range(len(phasearray)):
        phasearray[k] = pi_mod(phasearray[k])
    topcount = 0
    midcount = 0
    bottomcount = 0
    for k in range(len(phasearray)):
        if phasearray[k] < pi/2:
            bottomcount += 1
        elif phasearray[k] > 3*pi/2:
            topcount += 1
        else:
            midcount += 1
    if ((bottomcount + topcount) > midcount):
        mult = numpy.sign(bottomcount - topcount)
        if mult ==0: mult = 1
        cutoff = pi
        for k in range(len(phasearray)):
            if mult*phasearray[k] > mult*float(cutoff):
                phasearray[k] = phasearray[k] - mult*2*pi
    return phasearray

def pi_mod(pavg):
    while (pavg < 0 or pavg > 2*pi):
        if pavg < 0:
            pavg += 2*pi
        if pavg > 2*pi:
            pavg -= 2*pi
    return pavg

def writefunc(k):
    if k % 10 == 0:
    #if (k >= start1 and k <= stop1) or (k >= start2 and k <= stop2) or (k >= start3 and k <= stop3):
        return 1
    else:
        return 0

def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

def opt_freq(H, total_points, timestep, guess_freq):
    #Frequency from fft of whole dataset is not exactly correct.
    #Best results when fft is performed on a dataset with an integer number of periods
    #This routine tests the fft results of truncated versions of the dataset to find the optimal frequency
    #This procedure can take some time. The fft algorithm takes a variable amount of time depending on the
    #number of data points in the set. It is fastest (~1 second) when the number of data points is divisible by 2
    #and other small divisors. It is slowest when the number of data points is prime (~1 hour).
    #Thus I have implemented an algorithm to zero in on the optimal frequency by first
    #testing those truncated data sets with numbers of points divisble by powers of 2,
    #and then zeroing in from there.
    best_fom = 1
    #best_fom = 1000000000
    frequency = 0
    guess_period = 1/guess_freq
    guess_tsteps_in_period = int(guess_period/timestep)
    powersof2 = [1024, 256, 64, 16, 4]
    #powersof2 = [1024, 256, 64, 16, 4, 1]
    #powersof2 = [256, 64, 16, 4, 1]
    last_i =  int((total_points - guess_tsteps_in_period/2))
    jrange =  int(powersof2[0]*math.floor(guess_tsteps_in_period/(2*powersof2[0])))
    for j in range(len(powersof2)):
        freqsforfom = []
        fom = []
        ilist = []
        fundindexlist = []
        if j > 0:
            jrange = powersof2[j-1]
        center_i =  int(powersof2[j]*math.floor(last_i/powersof2[j]))
        lower_lim = max(center_i - jrange, int(3*guess_tsteps_in_period/4))
        upper_lim = min(center_i + jrange, total_points)
        print('center_i: '+str(center_i))
        print('jrange: '+str(jrange))
        for i in range(lower_lim, upper_lim, powersof2[j]):
            print(i)
#            if (numpoints % int(powersof2[j]) == 0) and (abs(i - last_i) < jrange):
            print('testing: '+str(i)+'/'+str(guess_tsteps_in_period))
            testH = H[0:(i)]
            print('length of testH: '+str(len(testH)))
            print('time: '+strftime('%H:%M:%S'))
            testHspectrum = numpy.fft.fft(testH)
            freqspectrum = numpy.fft.fftfreq((i), d=timestep)
            fundindex = numpy.argmax(numpy.abs(testHspectrum[1:(int(total_points/2))]))+1
            print('fundindex: '+str(fundindex))
            guess_freq = abs(freqspectrum[fundindex])
            print('guess_freq: '+str(guess_freq))
            offpeak = numpy.abs(testHspectrum[fundindex+1])
            onpeak = numpy.abs(testHspectrum[fundindex])
            fomi = offpeak/onpeak
#            fomi = testHspectrum[0]
            fom.append(fomi)
            freqsforfom.append(guess_freq)
            ilist.append(i)
            fundindexlist.append(fundindex)
            print('fom: '+str(fomi))
            if fomi < best_fom:
                best_fom = fomi
                frequency = guess_freq
                best_i = i
                last_i = best_i
                print('new optimal frequency = '+str(frequency))
        '''
        plt.figure(100)
        plt.style.use('classic')
        plt.title('FOM')
        plt.grid(True)
        plt.xlabel('i')
        plt.ylabel(r'fundindex')
        plt.xticks(rotation=90)
        plt.plot(ilist, fundindexlist, 'b.', label = r'M')
        #plt.ylim(0, 2*pi)
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()

        plt.figure(100)
        plt.style.use('classic')
        plt.title('FOM')
        plt.grid(True)
        plt.xlabel('i')
        plt.ylabel(r'FOM')
        plt.xticks(rotation=90)
        plt.plot(ilist, fom, 'b.', label = r'M')
        #plt.ylim(0, 2*pi)
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        #figoutfile = 'gmag.pdf'
        #plt.savefig(figoutfile)
        plt.show()
        '''
    return(frequency)
        
def fundmagphase(outdir, kstep, ambrellfile, descriptor, gfile, Hgfile, high_cutoff_freq, known_freq, MoverHforsubtraction, pMminuspHforsubtraction, MoverHforcalib, pMminuspHforphaseadj, MoverH0forsubtraction, est_num_periods):

    runoutfile = outdir+'\\'''+str(kstep)+descriptor+'.txt'
    runfile=open(runoutfile, 'a')
    
    ambrelldata = pd.read_csv(ambrellfile)
    
    
    times = flatten(ambrelldata.iloc[:,0:1].values.tolist())
#    print('times float: '+str(times[0:10]))
    M = flatten(ambrelldata.iloc[:,1:2].values.tolist())
#    print('M: '+str(M[0:10]))
    H = flatten(ambrelldata.iloc[:,2:3].values.tolist())
    for i in range(len(M)):
        M[i] = polarity*M[i]
    
    
    
    total_points = len(M)
    timestep = (times[total_points - 1]-times[0])/(total_points-1)
    print('timestep: '+str(timestep))


    
    bigHspectrum = numpy.fft.fft(H)
    #bigHspectrum = numpy.fft.fft(H)
    bigfreq = numpy.fft.fftfreq(total_points, d=timestep) 
    
    fundindex = numpy.argmax(numpy.abs(bigHspectrum[1:(int(total_points/2))]))+1
    guess_freq = bigfreq[fundindex]

    #Best results when exact frequency is used. The most accurate frequency
    #can be found using the opt_freq routine. However, it can take some time.
    #If the frequency is known from previous runs or because it is manufactured
    #test data, it should be entered in as one of the parameters of
    #fundmagphase. If the frequency is unknown and the user wants to run
    #opt_freq, input 'known_freq' as 0.
    if known_freq == 0:
        frequency = opt_freq(H, total_points, timestep, guess_freq)
    else:
        frequency = known_freq
    
    plt.figure(3)
    plt.style.use('classic')
    plt.title('M')
    plt.grid(True)
    plt.xlabel('data points')
    plt.ylabel(r'M')
    plt.xticks(rotation=90)
    plt.plot(range(total_points), M, 'b.', label = r'M')
    #plt.ylim(0, 2*pi)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()
    
    extra_points = 0
    
    
    period = 1/frequency
    tsteps_in_period = int(period/timestep)
    print('tsteps_in_period: '+str(tsteps_in_period))
#    tsteps_in_period = 11161    

#This section chooses the number of periods to analyze (an integer). It can be
#implemented as an interactive session with the user, or the number of
#periods can be entered as an input parameter (how it is currently set up).    
    print('Here is the Ambrell data. Choose a region you would like to analyze. The number of data poins per period is approximately '+str(tsteps_in_period))
    
    selectionproceed = 'n'
    k = 0
    while selectionproceed != 'y':
    
        #lower = input("Enter the data point numbers for the beginning of the reigon.")
        #lower = int(lower)
        #upper= input("Enter the data point numbers for the end of the reigon.")
        #upper = int(upper)
        
#        lower = 0
#        upper = 999999
        
#        est_num_periods = int(numpy.floor((upper - lower)/tsteps_in_period))

        lower = 0
        upper = est_num_periods*tsteps_in_period-1

        adj_total_points = est_num_periods*tsteps_in_period + extra_points
        new_upper = lower+adj_total_points

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
        plt.plot(range(lower, new_upper), M[lower:new_upper], 'g.', label = descriptor+' M selection')
        plt.plot(range(lower, new_upper), H[lower:new_upper], 'b.', label = descriptor+' H selection')
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
        #selectionproceed = input('Are you ready to proceed with this selection of data? y to proceed, n to enter a different selection.')
        selectionproceed = 'y'


    
    
    times = times[lower:new_upper]
    M = M[lower:new_upper]
    H = H[lower:new_upper]
    
    #for now
    
    
    '''
    data = pd.read_csv(outfile)
    data.head()
    times = data.iloc[:,0:1].values.tolist()
    M = data.iloc[:,1:2].values.tolist()
    H = data.iloc[:,2:3].values.tolist()
    FG = data.iloc[:,3:4].values.tolist()
    t0 = times[0]
    total_points = len(H)
    for k in range(len(H)):
        M[k] = float(M[k][0])
        H[k] = float(H[k][0])
        FG[k] = float(FG[k][0])
        times[k] = float(times[k][0])
    #Note, the timestep between each data point is actually slightly different(why? is it meaningful or just annoying?)
    '''

#FFT to create spectrum of truncated data        
    Mspectrum = numpy.fft.fft(M)
    Hspectrum = numpy.fft.fft(H)
    freq = numpy.fft.fftfreq(adj_total_points, d=timestep)        
    fftoutfile = 'fft.txt'
    fftfile=open(fftoutfile, 'a')
    #freq = scope.query('AFG:frequency?')
    #file2.write('AFG frequency = '+str(freq)+' Hz')
    fftfile.write('Frequency, Mspectrum, Hspectrum \n')
    for k in range(len(Mspectrum)):
        fftfile.write(str(freq[k])+',')
        fftfile.write(str(Mspectrum[k])+',')
        fftfile.write(str(Hspectrum[k]))
        fftfile.write('\n')

#Determine the frequency (again... should be redundant).
    halfpoints = int(adj_total_points/2)
    fundindex = numpy.argmax(numpy.abs(Hspectrum[1:halfpoints]))+1
    frequency = abs(freq[fundindex])
    period = 1/frequency
    tsteps_in_period = int(period/timestep)
    est_num_periods = fundindex
#Determine some basic info about the fundamental frequency (phase and mag)
    pH = numpy.angle(Hspectrum[est_num_periods])
    pM = numpy.angle(Mspectrum[est_num_periods])
    pMminuspH = pM - pH
    #anglediff[k] = (4*pi + numpy.angle(Mspectrum[num_per]) - numpy.angle(Hspectrum[num_per])) % (2*pi)
    Hmag = numpy.abs(Hspectrum[est_num_periods])
    Mmag = numpy.abs(Mspectrum[est_num_periods])
    MoverH = Mmag/Hmag    
    Hphasereal = numpy.cos(pH)
    Hphaseimag = numpy.sin(pH)
    MoverH0 = Mspectrum[0]/Hspectrum[0]
    if (pMminuspH > pi/2) or (pMminuspH < -pi/2):
        pMminuspH -= pi
        MoverH = -MoverH

    print('SPECTRUM SUMMARY: '+str(descriptor)+' spectrum. MoverH: '+str(MoverH)+', pMminuspH: '+str(pMminuspH))

    plt.figure(15+k*4)
    plt.style.use('classic')
    plt.title('Mspectrum before')
    plt.grid(True)
    plt.xlabel('frequency')
    plt.ylabel(r'Mspectrum')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.real(Mspectrum), 'b', label = 'real')
    plt.plot(freq, numpy.imag(Mspectrum), 'g', label = 'imag')
    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    plt.figure(16+k*4)
    plt.style.use('classic')
    plt.title('Mspectrum before')
    plt.grid(True)
    plt.xlabel('frequency')
    plt.ylabel(r'Mspectrum')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.real(Mspectrum), 'b', label = 'real')
    plt.plot(freq, numpy.imag(Mspectrum), 'g', label = 'imag')
    plt.xlim(-20000000, 20000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()


    print('BEFORE MANIPULATION')
    print('Mspectrum[est_num_periods]: '+str(Mspectrum[est_num_periods]))
    print('Mspectrum[3*est_num_periods]: '+str(Mspectrum[3*est_num_periods]))
    print('Mspectrum[5*est_num_periods]: '+str(Mspectrum[5*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-est_num_periods]: '+str(Mspectrum[len(Mspectrum)-est_num_periods]))
    print('Mspectrum[len(Mspectrum)-3*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-3*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-5*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-5*est_num_periods]))
    
    pMminuspHforsubtractionreal = numpy.cos(pMminuspHforsubtraction)
    pMminuspHforsubtractionimag = numpy.sin(pMminuspHforsubtraction)

#SUBTRACTION OF EMPTY SPECTRUM
    for i in [est_num_periods, (len(Mspectrum)-est_num_periods)]:
        print('i = '+str(i))
        print('Mspectrum[i] before subtraction: '+str(Mspectrum[i]))
        phase_sign = numpy.sign(freq[i])
        transfer_func_Hphase = complex(Hphasereal, -phase_sign*Hphaseimag)
        Mspectrum[i] = Mspectrum[i]*transfer_func_Hphase
        print('Mspectrum[i] after H transfer func: '+str(Mspectrum[i]))  
        print('MoverHforsubtraction: '+str(MoverHforsubtraction))
        print('Hmag: '+str(Hmag))
        print('pMminuspHforsubtractionreal: '+str(pMminuspHforsubtractionreal))
        Mspectrum[i] -= MoverHforsubtraction*Hmag*complex(pMminuspHforsubtractionreal, phase_sign*pMminuspHforsubtractionimag)
        print('Mspectrum[i] after direct subtraction: '+str(Mspectrum[i]))   
        Mspectrum[i] = Mspectrum[i]/transfer_func_Hphase
        print('Mspectrum[i] after subtraction and reverse H transfer func: '+str(Mspectrum[i]))
#    Mspectrum[0] -= MoverH0forsubtraction*Hspectrum[0]

    print('AFTER SUBTRACTION')
    print('Mspectrum[est_num_periods]: '+str(Mspectrum[est_num_periods]))
    print('Mspectrum[3*est_num_periods]: '+str(Mspectrum[3*est_num_periods]))
    print('Mspectrum[5*est_num_periods]: '+str(Mspectrum[5*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-est_num_periods]: '+str(Mspectrum[len(Mspectrum)-est_num_periods]))
    print('Mspectrum[len(Mspectrum)-3*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-3*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-5*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-5*est_num_periods]))

#SHIFT OF MSPECTRUM BY SAME AMOUNT AS EMPTY SPECTRUM
    '''
    for i in range(len(Mspectrum)):
        phase_sign_2 = numpy.sign(freq[i])
        if phase_sign_2 == 0.0: phase_sign_2 = 1.0
        adjpMminuspHforsubtraction = pMminuspHforsubtraction*numpy.abs(freq[i])/frequency
        adjpMminuspHforsubtractionreal = numpy.cos(adjpMminuspHforsubtraction)
        adjpMminuspHforsubtractionimag = numpy.sin(adjpMminuspHforsubtraction)
        transfer_func_pMminuspH = complex(adjpMminuspHforsubtractionreal, phase_sign_2*adjpMminuspHforsubtractionimag)
#        if (i!=0) and (abs(freq[i]) < high_cutoff_freq):# and  (i % est_num_periods == 0)
#        if i!=0 and abs(freq[i]) < high_cutoff_freq and  (i % est_num_periods == 0):
        if i>0:
            Mspectrum[i] = Mspectrum[i]*transfer_func_pMminuspH
#            transfer_func_g = complex(g_interp_real(freq[i]), phase_sign_2*g_interp_imag(freq[i]))
#            print('frequency: '+str(freq[i])+', transfer_func_Hphase: '+str(transfer_func_Hphase)+', transfer_func_g: '+str(transfer_func_g)+', transfer_func_pMminuspH: '+str(transfer_func_pMminuspH))
#            Mspectrum_gcorr[i] = Mspectrum_gcorr[i]*transfer_func_g
#                transfer_func_pMminuspH = complex(numpy.cos(pMminuspHforsubtraction), -phase_sign_2*numpy.sin(pMminuspHforsubtraction))
#                Mspectrum_gcorr[i] = Mspectrum_gcorr[i]*transfer_func_pMminuspH    

    print('AFTER EMPTY SPECTRUM PHASE ADJUSTMENT')
    print('Mspectrum[est_num_periods]: '+str(Mspectrum[est_num_periods]))
    print('Mspectrum[3*est_num_periods]: '+str(Mspectrum[3*est_num_periods]))
    print('Mspectrum[5*est_num_periods]: '+str(Mspectrum[5*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-est_num_periods]: '+str(Mspectrum[len(Mspectrum)-est_num_periods]))
    print('Mspectrum[len(Mspectrum)-3*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-3*est_num_periods]))
    print('Mspectrum[len(Mspectrum)-5*est_num_periods]: '+str(Mspectrum[len(Mspectrum)-5*est_num_periods]))
    '''

#G FACTOR CORRECTION

    g_interp_real, g_interp_imag = calculate_g(outdir, gfile, high_cutoff_freq)

    Hg_interp_real, Hg_interp_imag = calculate_Hg(outdir, Hgfile, high_cutoff_freq)

#    Hg_interp_real, Hg_interp_imag = calculate_Hg(gfile, frequency, high_cutoff_freq, pMminuspHforphaseadj)

    
    Mspectrum_gcorr = [0.0+0.0j]*len(Mspectrum)
    Hspectrum_gcorr = [0.0+0.0j]*len(Mspectrum)

#    pMminuspH_correction = pMminuspH_interp(frequency)
    
    #high_cutoff_freq = 300000
    
    for i in range(len(Mspectrum)):
#        phase_sign_2 = numpy.sign(freq[i])
#        if phase_sign_2 == 0.0: phase_sign_2 = 1.0
#        transfer_func_Hphase = complex(Hphasereal, -phase_sign_2*Hphaseimag)
#        if (i!=0) and (abs(freq[i]) < high_cutoff_freq):# and  (i % est_num_periods == 0)
#        if i!=0 and abs(freq[i]) < high_cutoff_freq and  (i % est_num_periods == 0):
        if (i>0) and (numpy.abs(freq[i]) < high_cutoff_freq) and odd_harmonic_M(len(freq), i, est_num_periods) == 1:
#        if (i>0) and (numpy.abs(freq[i]) < high_cutoff_freq) and i % (est_num_periods) == 0:
            Mspectrum_gcorr[i] = Mspectrum[i]
            phase_sign_2 = numpy.sign(freq[i])
            transfer_func_g = complex(g_interp_real(abs(freq[i])), phase_sign_2*g_interp_imag(abs(freq[i])))
#            transfer_func_g = 1
            Mspectrum_gcorr[i] = Mspectrum_gcorr[i]*transfer_func_g
        if i == est_num_periods or i == (len(Mspectrum) - est_num_periods):
            Hspectrum_gcorr[i] = Hspectrum[i]            
            phase_sign_2 = numpy.sign(freq[i])
            transfer_func_Hg = complex(Hg_interp_real(abs(freq[i])), phase_sign_2*Hg_interp_imag(abs(freq[i])))
#            transfer_func_Hg = 1
#            transfer_func_Hg = complex(numpy.cos(pMminuspHforphaseadj), phase_sign_2*numpy.sin(pMminuspHforphaseadj))
            Hspectrum_gcorr[i] = Hspectrum_gcorr[i]*transfer_func_Hg

    print('AFTER G-FACTOR WHICH INCLUDES PHASE ADJUSTMENT')
    print('Mspectrum_gcorr[est_num_periods]: '+str(Mspectrum_gcorr[est_num_periods]))
    print('Mspectrum_gcorr[3*est_num_periods]: '+str(Mspectrum_gcorr[3*est_num_periods]))
    print('Mspectrum_gcorr[5*est_num_periods]: '+str(Mspectrum_gcorr[5*est_num_periods]))
    print('Mspectrum_gcorr[len(Mspectrum)-est_num_periods]: '+str(Mspectrum_gcorr[len(Mspectrum)-est_num_periods]))
    print('Mspectrum_gcorr[len(Mspectrum)-3*est_num_periods]: '+str(Mspectrum_gcorr[len(Mspectrum)-3*est_num_periods]))
    print('Mspectrum_gcorr[len(Mspectrum)-5*est_num_periods]: '+str(Mspectrum_gcorr[len(Mspectrum)-5*est_num_periods]))
    print('AFTER G-FACTOR WHICH INCLUDES PHASE ADJUSTMENT')
    print('Hspectrum_gcorr[est_num_periods]: '+str(Hspectrum_gcorr[est_num_periods]))
    print('Hspectrum_gcorr[3*est_num_periods]: '+str(Hspectrum_gcorr[3*est_num_periods]))
    print('Hspectrum_gcorr[5*est_num_periods]: '+str(Hspectrum_gcorr[5*est_num_periods]))
    print('Hspectrum_gcorr[len(Mspectrum)-est_num_periods]: '+str(Hspectrum_gcorr[len(Mspectrum)-est_num_periods]))
    print('Hspectrum_gcorr[len(Mspectrum)-3*est_num_periods]: '+str(Hspectrum_gcorr[len(Mspectrum)-3*est_num_periods]))
    print('Hspectrum_gcorr[len(Mspectrum)-5*est_num_periods]: '+str(Hspectrum_gcorr[len(Mspectrum)-5*est_num_periods]))
    print('len(Mspectrum) = '+str(len(Mspectrum)))
    print('est_num_periods = '+str(est_num_periods))
    print('len(Mspectrum) = '+str(len(Mspectrum)))
    print('est_num_periods = '+str(est_num_periods))

    plt.figure(16+k*4)
    plt.style.use('classic')
    plt.title('Mspectrum after')
    plt.grid(True)
    plt.xlabel('frequency')
    plt.ylabel(r'Mspectrum')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.real(Mspectrum_gcorr), 'b', label = 'real')
    plt.plot(freq, numpy.imag(Mspectrum_gcorr), 'g', label = 'imag')
    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

#    print(Mspectrum_gcorr[0:10])

#    Mspectrum_gcorr = numpy.array(Mspectrum_gcorr)
 
#Reconstruction of signal
    Mreconstructed = numpy.fft.ifft(Mspectrum_gcorr)
    Hreconstructed = numpy.fft.ifft(Hspectrum_gcorr)

    pHg = numpy.angle(Hspectrum_gcorr[est_num_periods])
    pMg = numpy.angle(Mspectrum_gcorr[est_num_periods])
    pMminuspHg = pMg - pHg
    #anglediff[k] = (4*pi + numpy.angle(Mspectrum[num_per]) - numpy.angle(Hspectrum[num_per])) % (2*pi)
    Hmagg = numpy.abs(Hspectrum_gcorr[est_num_periods])
    Mmagg = numpy.abs(Mspectrum_gcorr[est_num_periods])
    MoverHg = Mmagg/Hmagg
    if (pMminuspHg > pi/2) or (pMminuspHg < -pi/2):
        pMminuspHg -= pi
        MoverHg = -MoverHg

    print('SPECTRUM SUMMARY: '+str(descriptor)+' spectrum, g-corrected. MoverH: '+str(MoverHg)+', pMminuspH: '+str(pMminuspHg))

    
#Integration of signal    
    Mspectrum_int = [0.0+0.0j]*len(Mspectrum_gcorr)
    Hspectrum_int = [0.0+0.0j]*len(Mspectrum_gcorr)
    
    for i in range(len(Mspectrum_gcorr)):
        if i > 0:
            Mspectrum_int[i] = (-1.0j)*Mspectrum_gcorr[i]/(2*pi*(freq[i]))
            Hspectrum_int[i] = (-1.0j)*Hspectrum_gcorr[i]/(2*pi*(freq[i]))

#Reconstruction of integrated signal
    Mintreconstructed = numpy.fft.ifft(Mspectrum_int)
    Hintreconstructed = numpy.fft.ifft(Hspectrum_int)

#Try integrating another way to check
    Mintreconstructed2 = [0.0]*len(Mreconstructed)
    Hintreconstructed2 = [0.0]*len(Mreconstructed)
    for i in range(len(Mreconstructed)-1):
        Mintreconstructed2[i+1] = Mintreconstructed2[i]+Mreconstructed[i]*timestep
        Hintreconstructed2[i+1] = Hintreconstructed2[i]+Hreconstructed[i]*timestep
    Mintreconstructed2[0] = Mintreconstructed2[1]-(Mintreconstructed2[2]-Mintreconstructed2[1])
    Hintreconstructed2[0] = Hintreconstructed2[1]-(Hintreconstructed2[2]-Hintreconstructed2[1])
    cM = numpy.mean(Mintreconstructed2)
    cH = numpy.mean(Hintreconstructed2)
    for i in range(len(Mreconstructed)):
        Mintreconstructed2[i] -= cM
        Hintreconstructed2[i] -= cH

#   M AND H CALIBRATION
    for i in range(len(Mintreconstructed)):
        Mintreconstructed[i] = M_calib_factor*Mintreconstructed[i]
        Mintreconstructed2[i] = M_calib_factor*Mintreconstructed2[i]
        Hintreconstructed[i] = H_calib_factor*Hintreconstructed[i]
        Hintreconstructed2[i] = H_calib_factor*Hintreconstructed2[i]


    plt.figure(10+k*4)
    plt.style.use('classic')
    plt.title('Mreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mreconstructed')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], M[0:22222], 'b.', label = descriptor+' M')
    plt.plot(times[0:22222], H[0:22222], 'g.', label = descriptor+' H')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    print('Mintreconstructed[0:10]: '+str(Mintreconstructed[0:10]))
    print('Hintreconstructed[0:10]: '+str(Hintreconstructed[0:10]))
        
    plt.figure(7+k*4)
    plt.style.use('classic')
    plt.title('Mreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mreconstructed')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], Mreconstructed[0:22222], 'b.', label = descriptor+' Mreconstructed')
    plt.plot(times[0:22222], Hreconstructed[0:22222], 'g.', label = descriptor+' Hreconstructed')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    plt.figure(7+k*4)
    plt.style.use('classic')
    plt.title('Mreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mreconstructed')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], Mintreconstructed[0:22222], 'b.', label = descriptor+' Mintreconstructed')
    plt.plot(times[0:22222], Hintreconstructed[0:22222], 'g.', label = descriptor+' Hintreconstructed')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    plt.figure(12+k*4)
    plt.style.use('classic')
    plt.title('Mreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mreconstructed')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], Mreconstructed[0:22222], 'b.', label = descriptor+' Mreconstructed')
    plt.plot(times[0:22222], M[0:22222], 'g.', label = descriptor+' M original')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    plt.figure(13+k*4)
    plt.style.use('classic')
    plt.title('Mintreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mintreconstructed 1 and 2')
    plt.xticks(rotation=90)
    plt.plot(times[0:22], Mintreconstructed2[0:22], 'b', label = descriptor+' Mintreconstructed2')
    plt.plot(times[0:22], Mintreconstructed[0:22], 'g', label = descriptor+' Mintreconstructed')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    '''
    plt.figure(14+k*4)
    plt.style.use('classic')
    plt.title('Mintreconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'Mintreconstructed 1 and 2')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], (Mintreconstructed2[0:22222]-Mintreconstructed[0:22222]), 'b', label = descriptor+'Difference Mintreconstructed2')
#    plt.plot(times[0:22222], Mintreconstructed[0:22222], 'g', label = descriptor+' Mintreconstructed')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    '''    
    
    plt.figure(12+k*4)
    plt.style.use('classic')
    plt.title('H reconstructed')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(r'H')
    plt.xticks(rotation=90)
    plt.plot(times[0:22222], Hreconstructed[0:22222], 'b.', label = descriptor+' Hreconstructed')
    plt.plot(times[0:22222], H[0:22222], 'g.', label = descriptor+' H original')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    '''
    plt.figure(9+k*4)
    plt.style.use('classic')
    plt.title('spectrum')
    plt.grid(True)
    plt.xlabel('freq')
    plt.ylabel(r'amplitude')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.abs(Mspectrum), 'b.', label = descriptor+' Mspectrum')
    #plt.plot(freq, numpy.abs(Hspectrum), 'g.', label = descriptor+' Hspectrum')
    #plt.ylim(0, 2*pi)
    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    
    plt.figure(8+k*4)
    plt.style.use('classic')
    plt.title('g-corrected spectrum')
    plt.grid(True)
    plt.xlabel('freq')
    plt.ylabel(r'amplitude')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.abs(Mspectrum_gcorr), 'b.', label = descriptor+' Mspectrum_gcorr')
    #plt.plot(freq, numpy.abs(Hspectrum_gcorr), 'g.', label = descriptor+' Hspectrum_gcorr')
    #plt.ylim(0, 2*pi)
    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()

    
    plt.figure(10+k*4)
    plt.style.use('classic')
    plt.title('spectrum for integration')
    plt.grid(True)
    plt.xlabel('freq')
    plt.ylabel(r'amplitude')
    plt.xticks(rotation=90)
    plt.plot(freq, numpy.abs(Mspectrum_int), 'b.', label = descriptor+' Mspectrum_int')
    #plt.plot(freq, numpy.abs(Hspectrum_int), 'g.', label = descriptor+' Hspectrum_int')
    #plt.ylim(0, 2*pi)
    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #figoutfile = 'gmag.pdf'
    #plt.savefig(figoutfile)
    plt.show()
    #plt.close()
    '''    
    plt.figure(9+k*4)
    plt.style.use('classic')
    plt.title('M vs H')
    plt.grid(True)
    plt.xlabel('H')
    plt.ylabel('M')
    plt.xticks(rotation=90)
    plt.plot(Hintreconstructed, Mintreconstructed, 'g.', label = r'M vs H')
    plt.plot(Hintreconstructed[0:3000], Mintreconstructed[0:3000], 'b.', label = descriptor+' M vs H first bit')
    plt.plot(Hintreconstructed[0:30], Mintreconstructed[0:30], 'm.', label = descriptor+' M vs H starting point')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = outdir+'\\'+str(kstep)+'-MH.pdf'
    plt.savefig(figoutfile)
    plt.show()
    #plt.close()  

    plt.figure(9+k*4)
    plt.style.use('classic')
    plt.title('M vs H')
    plt.grid(True)
    plt.xlabel('H')
    plt.ylabel('M')
    plt.xticks(rotation=90)
    plt.plot(Hintreconstructed2, Mintreconstructed2, 'g.', label = r'M vs H2')
    plt.plot(Hintreconstructed2[0:3000], Mintreconstructed2[0:3000], 'b.', label = descriptor+' M vs H first bit')
    plt.plot(Hintreconstructed2[0:30], Mintreconstructed2[0:30], 'm.', label = descriptor+' M vs H starting point')
    #plt.ylim(0, 2*pi)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = outdir+'\\'+str(kstep)+'-MH-2.pdf'
    plt.savefig(figoutfile)
    plt.show()
    #plt.close()  

#    inputs:
#    ambrellfile, descriptor, gfile, high_cutoff_freq, known_freq, MoverHforsubtraction, pMminuspHforsubtraction, pMminuspHforphaseadj,
# MoverH0forsubtraction, est_num_periods




#    filename = str(int(frequency))+'-'+str(j)

    runfile.write('INPUTS: \n')
    runfile.write('outdir = '+outdir+'\n')
    runfile.write('kstep = '+str(kstep)+'\n')
    runfile.write('ambrellfile = '+ambrellfile+'\n')
    runfile.write('descriptor = '+descriptor+'\n')
    runfile.write('gfile = '+gfile+'\n')
    runfile.write('high_cutoff_freq = '+str(high_cutoff_freq)+'\n')
    runfile.write('knownfreq = '+str(known_freq)+' (0 means run frequency optimization) \n')
    runfile.write('MoverHforsubtraction = '+str(MoverHforsubtraction)+'\n')
    runfile.write('pMminuspHforsubtraction = '+str(pMminuspHforsubtraction)+'\n')
    runfile.write('MoverHforcalib = '+str(MoverHforcalib)+'\n')
    runfile.write('pMminuspHforphaseadj = '+str(pMminuspHforphaseadj)+'\n')
    runfile.write('MoverH0forsubtraction = '+str(MoverH0forsubtraction)+'\n')
    runfile.write('est_num_periods = '+str(est_num_periods)+'\n')
    
    runfile.write('OUTPUTS: \n')
    runfile.write('Mintreconstructed... see file \n')
    runfile.write('Hintreconstructed... se file \n')
    runfile.write('MoverH = '+str(MoverH)+'\n')
    runfile.write('pMminuspH = '+str(pMminuspH)+'\n')
    runfile.write('MoverHg = '+str(MoverHg)+'\n')
    runfile.write('pMminuspHg = '+str(pMminuspHg)+'\n')
    runfile.write('MoverH0 = '+str(MoverH0)+'\n')

    kstep += 1
    return(Mintreconstructed, Hintreconstructed, MoverH, pMminuspH, MoverHg, pMminuspHg, MoverH0, kstep)


#frequency = input("First, type in the applied frequency in Hz (e.g. 224000). Or type quit to quit.")

#if frequency == 'quit':
#    sys.exit()

def calculate_g(outdir, gfile, high_cutoff_freq):
    #Read in data for g-factor determination
    gdata = pd.read_csv(gfile)
    avgfreqlist = flatten(gdata.iloc[:,0:1].values.tolist())
#    avgfreqerrlist = flatten(avgdata.iloc[:,1:2].values.tolist())
    greallist = flatten(gdata.iloc[:,7:8].values.tolist())
    gimaglist = flatten(gdata.iloc[:,8:9].values.tolist())
#    pMminuspHlist = flatten(avgdata.iloc[:,6:7].values.tolist())
#    pMminuspHerrlist = flatten(avgdata.iloc[:,7:8].values.tolist())
    #avgFGlist = flatten(avgdata.iloc[:,8:9].values.tolist())
    #avgFGerrlist = flatten(avgdata.iloc[:,9:10].values.tolist())
    #avgHlist = flatten(avgdata.iloc[:,10:11].values.tolist())
    #avgHerrlist = flatten(avgdata.iloc[:,11:12].values.tolist())
    #avgMlist = flatten(avgdata.iloc[:,12:13].values.tolist())
    #avgMerrlist = flatten(avgdata.iloc[:,13:14].values.tolist())
    #normHlist = flatten(avgdata.iloc[:,14:15].values.tolist())
    #normHerrlist = flatten(avgdata.iloc[:,15:16].values.tolist())
    #normMlist = flatten(avgdata.iloc[:,16:17].values.tolist())
    #normMerrlist = flatten(avgdata.iloc[:,17:18].values.tolist())
    


    
#    pMminuspH_interp = interp1d(avgfreqlist, pMminuspHlist, kind='cubic', assume_sorted=False)
#    pMminuspH_fund = pMminuspH_interp(frequency)
#    print('pMminuspH_interp value: '+str(pMminuspH_fund))

    

    '''    
    gphase_at_freq = gphase_interp(frequency)
    phase_corr = pMminuspHg + pi_mod(gphase_at_freq)
    print('pMminuspH for subtraction =: '+str(pMminuspH))
    print('gphase_interp(frequency): '+str(gphase_at_freq))
    print('phase_corr = pMminuspH + gphase_interp(frequency): '+str(phase_corr))

    print('Phase correction of g-factor by pMminuspH: ')
    for i in range(len(avgfreqlist)):
        print('frequency: '+str(avgfreqlist[i]))
        print('gphase[i] before phasecorr: '+str(gphase[i]))
        gphase[i] -= phase_corr*avgfreqlist[i]/frequency
        print('phase_corr*avgfreqlist[i]/frequency: '+str(phase_corr*avgfreqlist[i]/frequency))
        print('gphase[i] after phasecorr: '+str(gphase[i]))
        greal[i] = gmag[i]*numpy.cos(gphase[i])
        gimag[i] = gmag[i]*numpy.sin(gphase[i])
    print('normMlist[0]: '+str(normMlist[0]))
    gphase = pi_mod_array(gphase)

    gphase_interp = interp1d(avgfreqlist, gphase, kind='cubic', assume_sorted=False)    
     
    g1test = ['?']*len(avgfreqlist)
    for i in range(len(avgfreqlist)):
        #gimaglist[i] = -gimaglist[i]
        print('i = '+str(i))
        g1test[i] = numpy.sqrt(greallist[i]*greallist[i]+gimaglist[i]*gimaglist[i])
    '''
    high_cutoff_freq = numpy.amin([high_cutoff_freq, avgfreqlist[-1]])
    
    print('len(avgfreqlist) = '+str(len(avgfreqlist)))
    print('len(greallist) = '+str(len(greallist)))

    g_interp_real = interp1d(avgfreqlist, greallist, kind='linear', assume_sorted = False)
    g_interp_imag = interp1d(avgfreqlist, gimaglist, kind='linear', assume_sorted = False)
       
    plt.figure(2)
    plt.style.use('classic')
    plt.title('gcomplex')
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'g')
    plt.xticks(rotation=90)
    plt.plot(avgfreqlist, greallist, 'b', label = r'g_real')
    plt.plot(avgfreqlist, gimaglist, 'g', label = r'g_imag')
#    plt.plot(avgfreqlist, g1test, 'm', label = 'g1test')
#    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = outdir+'\\gcomplex.pdf'
    plt.savefig(figoutfile)
    plt.show()
    plt.close()
       
    #print('Do the g files look okay?')
    
    #qproceed = input("Press any key when you are ready to continue, or type quit to quit.")
    
    #if qproceed == 'quit':
    #    sys.exit()
    return(g_interp_real, g_interp_imag)
    
    
def calculate_Hg(outdir, Hgfile, high_cutoff_freq):
    #Read in data for g-factor determination
    gdata = pd.read_csv(Hgfile)
    avgfreqlist = flatten(gdata.iloc[:,0:1].values.tolist())
#    avgfreqerrlist = flatten(avgdata.iloc[:,1:2].values.tolist())
    greallist = flatten(gdata.iloc[:,7:8].values.tolist())
    gimaglist = flatten(gdata.iloc[:,8:9].values.tolist())
#    pMminuspHlist = flatten(avgdata.iloc[:,6:7].values.tolist())
#    pMminuspHerrlist = flatten(avgdata.iloc[:,7:8].values.tolist())
    #avgFGlist = flatten(avgdata.iloc[:,8:9].values.tolist())
    #avgFGerrlist = flatten(avgdata.iloc[:,9:10].values.tolist())
    #avgHlist = flatten(avgdata.iloc[:,10:11].values.tolist())
    #avgHerrlist = flatten(avgdata.iloc[:,11:12].values.tolist())
    #avgMlist = flatten(avgdata.iloc[:,12:13].values.tolist())
    #avgMerrlist = flatten(avgdata.iloc[:,13:14].values.tolist())
    #normHlist = flatten(avgdata.iloc[:,14:15].values.tolist())
    #normHerrlist = flatten(avgdata.iloc[:,15:16].values.tolist())
    #normMlist = flatten(avgdata.iloc[:,16:17].values.tolist())
    #normMerrlist = flatten(avgdata.iloc[:,17:18].values.tolist())
    


    
#    pMminuspH_interp = interp1d(avgfreqlist, pMminuspHlist, kind='cubic', assume_sorted=False)
#    pMminuspH_fund = pMminuspH_interp(frequency)
#    print('pMminuspH_interp value: '+str(pMminuspH_fund))

    

    '''    
    gphase_at_freq = gphase_interp(frequency)
    phase_corr = pMminuspHg + pi_mod(gphase_at_freq)
    print('pMminuspH for subtraction =: '+str(pMminuspH))
    print('gphase_interp(frequency): '+str(gphase_at_freq))
    print('phase_corr = pMminuspH + gphase_interp(frequency): '+str(phase_corr))

    print('Phase correction of g-factor by pMminuspH: ')
    for i in range(len(avgfreqlist)):
        print('frequency: '+str(avgfreqlist[i]))
        print('gphase[i] before phasecorr: '+str(gphase[i]))
        gphase[i] -= phase_corr*avgfreqlist[i]/frequency
        print('phase_corr*avgfreqlist[i]/frequency: '+str(phase_corr*avgfreqlist[i]/frequency))
        print('gphase[i] after phasecorr: '+str(gphase[i]))
        greal[i] = gmag[i]*numpy.cos(gphase[i])
        gimag[i] = gmag[i]*numpy.sin(gphase[i])
    print('normMlist[0]: '+str(normMlist[0]))
    gphase = pi_mod_array(gphase)

    gphase_interp = interp1d(avgfreqlist, gphase, kind='cubic', assume_sorted=False)    
     
    g1test = ['?']*len(avgfreqlist)
    for i in range(len(avgfreqlist)):
        #gimaglist[i] = -gimaglist[i]
        print('i = '+str(i))
        g1test[i] = numpy.sqrt(greallist[i]*greallist[i]+gimaglist[i]*gimaglist[i])
    '''
    high_cutoff_freq = numpy.amin([high_cutoff_freq, avgfreqlist[-1]])
    
    print('len(avgfreqlist) = '+str(len(avgfreqlist)))
    print('len(greallist) = '+str(len(greallist)))

    Hg_interp_real = interp1d(avgfreqlist, greallist, kind='linear', assume_sorted = False)
    Hg_interp_imag = interp1d(avgfreqlist, gimaglist, kind='linear', assume_sorted = False)
       
    plt.figure(2)
    plt.style.use('classic')
    plt.title('gcomplex')
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'g')
    plt.xticks(rotation=90)
    plt.plot(avgfreqlist, greallist, 'b', label = r'g_real')
    plt.plot(avgfreqlist, gimaglist, 'g', label = r'g_imag')
#    plt.plot(avgfreqlist, g1test, 'm', label = 'g1test')
#    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = outdir+'\\gcomplex.pdf'
    plt.savefig(figoutfile)
    plt.show()
    plt.close()
       
    #print('Do the g files look okay?')
    
    #qproceed = input("Press any key when you are ready to continue, or type quit to quit.")
    
    #if qproceed == 'quit':
    #    sys.exit()
    return(Hg_interp_real, Hg_interp_imag)

'''
def calculate_Hg(gfile, frequency, high_cutoff_freq, pMminuspHg):
    #Read in data for g-factor determination
    avgdata = pd.read_csv(avgangledifffile)
    avgfreqlist = flatten(avgdata.iloc[:,0:1].values.tolist())
    avgfreqerrlist = flatten(avgdata.iloc[:,1:2].values.tolist())
    pHminuspFGlist = flatten(avgdata.iloc[:,2:3].values.tolist())
    pHminuspFGerrlist = flatten(avgdata.iloc[:,3:4].values.tolist())
#    pMminuspHlist = flatten(avgdata.iloc[:,6:7].values.tolist())
#    pMminuspHerrlist = flatten(avgdata.iloc[:,7:8].values.tolist())
    #avgFGlist = flatten(avgdata.iloc[:,8:9].values.tolist())
    #avgFGerrlist = flatten(avgdata.iloc[:,9:10].values.tolist())
    #avgHlist = flatten(avgdata.iloc[:,10:11].values.tolist())
    #avgHerrlist = flatten(avgdata.iloc[:,11:12].values.tolist())
    #avgMlist = flatten(avgdata.iloc[:,12:13].values.tolist())
    #avgMerrlist = flatten(avgdata.iloc[:,13:14].values.tolist())
    normHlist = flatten(avgdata.iloc[:,14:15].values.tolist())
    normHerrlist = flatten(avgdata.iloc[:,15:16].values.tolist())    
    
    
    delfreqlist = []
    for i in range(len(normHlist)):
        if (abs(normHerrlist[i]) > 0.07*abs(normHlist[i])):
            delfreqlist.append(i)
    for i in list(reversed(range(len(delfreqlist)))):
        del avgfreqlist[delfreqlist[i]]
        del avgfreqerrlist[delfreqlist[i]]
        del normHerrlist[delfreqlist[i]]
        del normHlist[delfreqlist[i]]
        del pHminuspFGlist[delfreqlist[i]]
        del pHminuspFGerrlist[delfreqlist[i]]

    Hgmag = ['?']*len(avgfreqlist)
    Hgphase = ['?']*len(avgfreqlist)
    Hgreal = ['?']*len(avgfreqlist)
    Hgimag = ['?']*len(avgfreqlist)
    
    for i in range(len(avgfreqlist)):
#        gmag[i] = normMlist[0]/normMlist[i]
        Hgmag[i] = 0.007/normHlist[i]
#        gphase[i] = 2*pi-(pMminuspFGlist[i])-pMminuspHg*avgfreqlist[i]/frequency
        Hgphase[i] = 2*pi-(pHminuspFGlist[i])
        Hgreal[i] = Hgmag[i]*numpy.cos(Hgphase[i])
        Hgimag[i] = Hgmag[i]*numpy.sin(Hgphase[i])
    print('normHlist[0]: '+str(normHlist[0]))
    Hgphase = pi_mod_array(Hgphase)
    
    Hgmag_interp = interp1d(avgfreqlist, Hgmag, kind='linear', assume_sorted=False)
    Hgphase_interp = interp1d(avgfreqlist, Hgphase, kind='linear', assume_sorted=False)
    
    Hgreal_interp = interp1d(avgfreqlist, Hgreal, kind='linear', assume_sorted = False)
    Hgimag_interp = interp1d(avgfreqlist, Hgimag, kind='linear', assume_sorted = False)

    ginterp_freq_for_plot = numpy.arange(avgfreqlist[0], avgfreqlist[-1], avgfreqlist[-1]/200)
    ginterp_mag_for_plot = Hgmag_interp(ginterp_freq_for_plot)
    ginterp_phase_for_plot = Hgphase_interp(ginterp_freq_for_plot)
    
    high_cutoff_freq = numpy.amin([high_cutoff_freq, avgfreqlist[-1]])

    Hg_freq = [0]*len(avgfreqlist)*2
    Hg_real = [0]*len(avgfreqlist)*2
    Hg_imag = [0]*len(avgfreqlist)*2 
    for i in range(len(avgfreqlist)):
        Hg_freq[len(avgfreqlist) + i] = avgfreqlist[i]
        Hg_real[len(avgfreqlist) + i] = Hgreal[i]
        Hg_imag[len(avgfreqlist) + i] = Hgimag[i]
        Hg_freq[len(avgfreqlist) - i-1] = - avgfreqlist[i]
        Hg_real[len(avgfreqlist) -i-1] = Hgreal[i]
        Hg_imag[len(avgfreqlist)-i-1] = Hgimag[i]
    Hg_interp_real = interp1d(Hg_freq, Hg_real, kind='linear', assume_sorted = False)
    Hg_interp_imag = interp1d(Hg_freq, Hg_imag, kind='linear', assume_sorted = False)
        
    plt.figure(1)
    plt.style.use('classic')
    plt.title('gmag')
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'Hgmag')
    plt.xticks(rotation=90)
    plt.plot(avgfreqlist, Hgmag, 'bo', label = r'Hgmag')
    plt.plot(ginterp_freq_for_plot, ginterp_mag_for_plot, 'g', label = r'interp Hgmag')
#    plt.xlim(-1000000, 1000000)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = 'gmag.pdf'
    plt.savefig(figoutfile)
    plt.show()
    plt.close()
    plt.figure(2)
    plt.style.use('classic')
    plt.title('gphase')
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'Hgphase')
    plt.xticks(rotation=90)
    plt.plot(avgfreqlist, Hgphase, 'bo', label = r'Hgphase')
    plt.plot(ginterp_freq_for_plot, ginterp_phase_for_plot, 'g', label = r'interp Hgphase')
#    plt.xlim(-1000000, 1000000)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = 'gphase.pdf'
    plt.savefig(figoutfile)
    plt.show()
    plt.close()
               
    plt.figure(2)
    plt.style.use('classic')
    plt.title('Hgcomplex')
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'Hg')
    plt.xticks(rotation=90)
    plt.plot(Hg_freq, Hg_real, 'b', label = r'Hg_real')
    plt.plot(Hg_freq, Hg_imag, 'g', label = r'Hg_imag')
#    plt.xlim(-1000000, 1000000)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    figoutfile = 'Hgcomplex.pdf'
    plt.savefig(figoutfile)
    plt.show()
    plt.close()
       
    #print('Do the g files look okay?')
    
    #qproceed = input("Press any key when you are ready to continue, or type quit to quit.")
    
    #if qproceed == 'quit':
    #    sys.exit()
    return(Hg_interp_real, Hg_interp_imag)
'''