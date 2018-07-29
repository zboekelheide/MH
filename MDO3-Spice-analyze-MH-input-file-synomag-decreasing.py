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

import JMDO3SpiceanalyzeMHdecreasing
#this file allows subfundamental frequencies

import MDO3SpiceanalyzeMH

#Stuff for doing fft analysis etc
pi = math.pi 
inf = numpy.inf

#high_cutoff_freq should be at least 3Mhz (3000000)
high_cutoff_freq = 8000000
figoutfile = 'synomag-MH-decreasing.pdf'
figoutfile2 = 'synomag-MH-decreasing-2.pdf'
tiffoutfile1 = 'synomag-MH-decreasing.tiff'
tiffoutfileslp = 'synomag-SLP.tiff'

basedir=r'''C:\Users\boekelhz\a1Lafayette\data\TektronixMDO3024\pythonMH\20180705'''
os.chdir(basedir)
datesubf = strftime("%Y%m%d")
timesubf = strftime("%H%M%S")
kstep=0
outdir = basedir+'\\20180703\\'''+datesubf+'\\'''+timesubf
if not os.path.exists(outdir):
    print('Creating directory')
    os.makedirs(outdir)  


gfile1MOhm = '20180621-gsim-1MOhm-formatted.csv'
Hgfile1MOhm = '20180627-Hgsim-1MOhm-formatted.csv'

ambrellemptyfile1MOhm = '145848\\empty-0.txt'


startnum = 4
endnum = 45
nullfilelist = [8, 11, 42, 43, 44]
synomagfile1MOhmlist = []
for i in range(startnum, endnum):
#    synomagfile1MOhmlist[i] = '163149\synomag-w-FO-'+str(k)+'.txt'
    if i in nullfilelist:
        continue
    else:
        k = 0
        fileexists = 1
        while fileexists == 1:
            filenm = '150020\\dataselections\\synomag-w-FO-decreasing-'+str(i)+'.txt'+str(k)
            if os.path.exists(filenm):
                synomagfile1MOhmlist.append('150020\\dataselections\\synomag-w-FO-decreasing-'+str(i)+'.txt'+str(k))
                k+=1
            else:
                fileexists = 0

numfiles = len(synomagfile1MOhmlist)            

cFe = 14200
#(in g(Fe)/m^3)

#gfile50Ohm = '20180621-gsim-50Ohm-formatted.csv'
#Hgfile50Ohm = '20180627-Hgsim-50Ohm-formatted.csv'
#ambrellemptyfile50Ohm = '122559\empty-zeroed-0.txt'
#ambrellstandardfile50Ohm = '143014-standard\standard-50Ohm.txt'
#ambrellsamplefile50Ohm = '122737\sample-zeroed-0.txt'
#ambrellsamplefile50Ohm2 = '122856\sample-zeroed2-0.txt'

#gfile75Ohm = '20180621-gsim-75Ohm-formatted.csv'
#Hgfile75Ohm = '20180627-Hgsim-75Ohm-formatted.csv'
#ambrellemptyfile75Ohm = '141033-empty\empty-75Ohm.txt'
#ambrellstandardfile75Ohm = '142826-standard\standard-75Ohm.txt'
#ambrellsamplefile75Ohm = '142104-sample\sample-75Ohm.txt'

#emptyambrelltestfile = 'test-with-sines-phaseoffset-empty.csv'
#fullambrelltestfile = 'test-with-sines-phaseoffsetplusnonlinearsample-full.csv'

#define multiplier (e.g. +1 or -1 depending on polarity)
#this is polarity like if one of the channels is inverted wrt the other
#polarity = 1

#does g factor make phase difference a lead or lag?
#phase_sign = -1

#FORMAT FOR OUTPUTS AND INPUTS:

# (Mintreconstructed,
# Hintreconstructed,
# MoverH,
# pMminuspH,
# pMminuspHg,
# MoverH0 (currently unused),
# kstep
# ) = MDO3SpiceanalyzeMH.fundmagphase(
#       outdir,
#       kstep,
#       ambrellfile,
#       descriptor (string),
#       gfile,
#       high_cutoff_freq,
#       known_freq,
#       MoverHforsubtraction,
#       pMminuspHforsubtraction,
#       pMminuspHforphaseadj,
#       MoverH0forsubtraction (currently unused),
#       standardMoverH,
#       est_num_periods)


####################################
# 50 OHM DATA
'''
(emptyMintreconstructed, 
 emptyHintreconstructed, 
 emptyMoverH, 
 emptypMminuspH, 
 emptyMoverHg,
 emptypMminuspHg, 
 emptyMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile50Ohm, 
         'empty 50Ohm', 
         gfile50Ohm,
         Hgfile50Ohm,
         high_cutoff_freq, 
         225213, 
         0, 
         0, 
         1, 
         0, 
         0,
         80
         )

(phasezeroedMintreconstructed,  #Mintreconstructed
 phasezeroedHintreconstructed,  #Hintreconstructed
 phasezeroedMoverH,             #MoverH before subtraction etc 
 phasezeroedpMminuspH,          #pMminuspH before subtraction etc
 phasezeroedMoverHg,            #MoverHg after subtraction and g-correction
 phasezeroedpMminuspHg,         #pMminuspHg after subtraction and g-correction
 phasezeroedMoverH0,            #MoverH0, currently not used
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir,                #outdir
         kstep,                 #kstep, to keep track of which run in this list of runs it is, for file naming purposes
         ambrellemptyfile50Ohm, #ambrell file
         'empty 50Ohm',         #descriptor: a string to describe this run
         gfile50Ohm,            #gfile
         Hgfile50Ohm,           #Hgfile
         high_cutoff_freq,      #high_cutoff_freq: Maximum frequency included in signal
         225158,                #known_freq: Known frequency of signal file. If unknown, input 0 to initiate frequency optimization routine
         0,                     #MoverHforsubtraction: MoverH of a previous run (presumably empty run), for subtraction of empty signal
         emptypMminuspH,        #pMminuspHforsubtraction: pMminuspH of a previous run before g-correction, for subtraction of empty signal
         1,                     #MoverHforcalib: for M calibration. Use 1 if M calibration has not been performed yet
         emptypMminuspHg,       #pMminuspHforphaseadj: pMminuspH of a previous run after g-correction, for use in phase adjustment
         emptyMoverH0,          #MoverH0forsubtraction: currently not used, may be used later
         8                      #est_num_periods: number of periods to analyze
         )
 
(zeroedMintreconstructed, 
 zeroedHintreconstructed, 
 zeroedMoverH, 
 zeroedpMminuspH,
 zeroedMoverHg, 
 zeroedpMminuspHg, 
 zeroedMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile50Ohm, 
         'empty 50Ohm', 
         gfile50Ohm, 
         Hgfile50Ohm,
         high_cutoff_freq, 
         225158, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

(standardMintreconstructed, 
 standardHintreconstructed, 
 standardMoverH, 
 standardpMminuspH,
 standardMoverHg, 
 standardpMminuspHg, 
 standardMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellstandardfile50Ohm, 
         'standard 50Ohm', 
         gfile50Ohm,
         Hgfile50Ohm,
         high_cutoff_freq, 
         225162, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         80
         )

(sample50OhmMintreconstructed, 
 sample50OhmHintreconstructed, 
 sampleMoverH, 
 samplepMminuspH,
 sampleMoverHg, 
 samplepMminuspHg, 
 sampleMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellsamplefile50Ohm, 
         'sample 50Ohm', 
         gfile50Ohm, 
         Hgfile50Ohm,
         high_cutoff_freq, 
         225166, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

(sample50OhmMintreconstructed2, 
 sample50OhmHintreconstructed2, 
 sampleMoverH, 
 samplepMminuspH,
 sampleMoverHg, 
 samplepMminuspHg, 
 sampleMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellsamplefile50Ohm2, 
         'sample 50Ohm', 
         gfile50Ohm, 
         Hgfile50Ohm,
         high_cutoff_freq, 
         225184, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

##############################################


##############################################


# 75 OHM DATA

(emptyMintreconstructed, 
 emptyHintreconstructed, 
 emptyMoverH, 
 emptypMminuspH, 
 emptyMoverHg,
 emptypMminuspHg, 
 emptyMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile75Ohm, 
         'empty 75Ohm', 
         gfile75Ohm,
         Hgfile75Ohm,
         high_cutoff_freq, 
         225136, 
         0, 
         0, 
         1,
         0, 
         0, 
         8
         )

(phasezeroedMintreconstructed,  #Mintreconstructed
 phasezeroedHintreconstructed,  #Hintreconstructed
 phasezeroedMoverH,             #MoverH before subtraction etc 
 phasezeroedpMminuspH,          #pMminuspH before subtraction etc
 phasezeroedMoverHg,            #MoverHg after subtraction and g-correction
 phasezeroedpMminuspHg,         #pMminuspHg after subtraction and g-correction
 phasezeroedMoverH0,            #MoverH0, currently not used
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir,                #outdir
         kstep,                 #kstep, to keep track of which run in this list of runs it is, for file naming purposes
         ambrellemptyfile75Ohm, #ambrell file
         'empty 75Ohm',         #descriptor: a string to describe this run
         gfile75Ohm,            #gfile
         Hgfile75Ohm,           #Hgfile
         high_cutoff_freq,      #high_cutoff_freq: Maximum frequency included in signal
         225136,                #known_freq: Known frequency of signal file. If unknown, input 0 to initiate frequency optimization routine
         0,                     #MoverHforsubtraction: MoverH of a previous run (presumably empty run), for subtraction of empty signal
         emptypMminuspH,        #pMminuspHforsubtraction: pMminuspH of a previous run before g-correction, for subtraction of empty signal
         1,                     #MoverHforcalib: for M calibration. Use 1 if M calibration has not been performed yet
         emptypMminuspHg,       #pMminuspHforphaseadj: pMminuspH of a previous run after g-correction, for use in phase adjustment
         emptyMoverH0,          #MoverH0forsubtraction: currently not used, may be used later
         8                      #est_num_periods: number of periods to analyze
         )

(zeroedMintreconstructed, 
 zeroedHintreconstructed, 
 zeroedMoverH, 
 zeroedpMminuspH, 
 zeroedMoverHg,
 zeroedpMminuspHg, 
 zeroedMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile75Ohm, 
         'empty 75Ohm', 
         gfile75Ohm, 
         Hgfile75Ohm,
         high_cutoff_freq, 
         225136, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

(standardMintreconstructed, 
 standardHintreconstructed, 
 standardMoverH, 
 standardpMminuspH, 
 standardMoverHg,
 standardpMminuspHg, 
 standardMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellstandardfile75Ohm, 
         'standard 75Ohm', 
         gfile75Ohm, 
         Hgfile75Ohm,
         high_cutoff_freq, 
         225201, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         80
         )

(sample75OhmMintreconstructed, 
 sample75OhmHintreconstructed, 
 sampleMoverH, 
 samplepMminuspH, 
 sampleMoverHg,
 samplepMminuspHg, 
 sampleMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellsamplefile75Ohm, 
         'sample 75Ohm', 
         gfile75Ohm, 
         Hgfile75Ohm,
         high_cutoff_freq, 
         225144, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

##############################################

'''
# 1 MOHM DATA

(emptyMintreconstructed, 
 emptyHintreconstructed, 
 emptyMoverH, 
 emptypMminuspH, 
 emptyMoverHg,
 emptypMminuspHg, 
 emptyMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile1MOhm, 
         'empty 1MOhm', 
         gfile1MOhm, 
         Hgfile1MOhm,
         high_cutoff_freq, 
         225198, 
         0, 
         0, 
         1,
         0, 
         0, 
         7
         )
'''
(phasezeroedMintreconstructed,  #Mintreconstructed
 phasezeroedHintreconstructed,  #Hintreconstructed
 phasezeroedMoverH,             #MoverH before subtraction etc 
 phasezeroedpMminuspH,          #pMminuspH before subtraction etc
 phasezeroedMoverHg,            #MoverHg after subtraction and g-correction
 phasezeroedpMminuspHg,         #pMminuspHg after subtraction and g-correction
 phasezeroedMoverH0,            #MoverH0, currently not used
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir,                #outdir
         kstep,                 #kstep, to keep track of which run in this list of runs it is, for file naming purposes
         ambrellemptyfile1MOhm, #ambrell file
         'empty 1MOhm',         #descriptor: a string to describe this run
         gfile1MOhm,            #gfile
         Hgfile75Ohm,
         high_cutoff_freq,      #high_cutoff_freq: Maximum frequency included in signal
         225198,                #known_freq: Known frequency of signal file. If unknown, input 0 to initiate frequency optimization routine
         0,                     #MoverHforsubtraction: MoverH of a previous run (presumably empty run), for subtraction of empty signal
         emptypMminuspH,        #pMminuspHforsubtraction: pMminuspH of a previous run before g-correction, for subtraction of empty signal
         1,                     #MoverHforcalib: for M calibration. Use 1 if M calibration has not been performed yet
         emptypMminuspHg,       #pMminuspHforphaseadj: pMminuspH of a previous run after g-correction, for use in phase adjustment
         emptyMoverH0,          #MoverH0forsubtraction: currently not used, may be used later
         8                      #est_num_periods: number of periods to analyze
         )

(zeroedMintreconstructed, 
 zeroedHintreconstructed, 
 zeroedMoverH, 
 zeroedpMminuspH, 
 zeroedMoverHg,
 zeroedpMminuspHg, 
 zeroedMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellemptyfile1MOhm, 
         'empty 1MOhm', 
         gfile1MOhm, 
         Hgfile75Ohm,
         high_cutoff_freq, 
         225198, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         8
         )

(standardMintreconstructed, 
 standardHintreconstructed, 
 standardMoverH, 
 standardpMminuspH,
 standardMoverHg, 
 standardpMminuspHg, 
 standardMoverH0, 
 kstep
 ) = MDO3SpiceanalyzeMH.fundmagphase(
         outdir, 
         kstep, 
         ambrellstandardfile1MOhm, 
         'standard 1MOhm', 
         gfile1MOhm, 
         Hgfile75Ohm,
         high_cutoff_freq, 
         225169, 
         emptyMoverH, 
         emptypMminuspH, 
         1,
         emptypMminuspHg, 
         emptyMoverH0, 
         80
         )
'''
synomagMintreconstructedlist = [[0,0]]*numfiles
synomagHintreconstructedlist = [[0,0]]*numfiles
hmax = [0] * numfiles
slp = [0] * numfiles
for i in range(numfiles):
#    if (i % 5) == 0:
    k = i
    print('filename = '+str(synomagfile1MOhmlist[i]))
    #sz = 0
    #sz = os.path.getsize(synomagfile1MOhmlist[i])
    #numcycles = int(numpy.floor(sz/250000))
    #if k not in nullfilelist:
    (synomagMintreconstructedlist[i], 
     synomagHintreconstructedlist[i], 
     sampleMoverH, 
     samplepMminuspH, 
     sampleMoverHg,
     samplepMminuspHg, 
     sampleMoverH0, 
     kstep,
     hmax[i],
     slp[i]
     ) = JMDO3SpiceanalyzeMHdecreasing.fundmagphase(
             outdir, 
             kstep, 
             synomagfile1MOhmlist[i], 
             'synomag '+str(k), 
             gfile1MOhm, 
             Hgfile1MOhm,
             high_cutoff_freq, 
             0, 
             emptyMoverH, 
             emptypMminuspH, 
             1,
             emptypMminuspHg, 
             emptyMoverH0, 
             2
             )
hmaxmax = numpy.amax(hmax)
'''
    if k > 22 and k != 8:
        (synomagMintreconstructedlist[i], 
         synomagHintreconstructedlist[i], 
         sampleMoverH, 
         samplepMminuspH, 
         sampleMoverHg,
         samplepMminuspHg, 
         sampleMoverH0, 
         kstep
         ) = MDO3SpiceanalyzeMH.fundmagphase(
                 outdir, 
                 kstep, 
                 synomagfile1MOhmlist[i], 
                 'synomag '+str(k), 
                 gfile1MOhm, 
                 Hgfile1MOhm,
                 high_cutoff_freq, 
                 0, 
                 emptyMoverH, 
                 emptypMminuspH, 
                 1,
                 emptypMminuspHg, 
                 emptyMoverH0, 
                 numcycles
                 )

    '''
'''




#emptyMintreconstructed, emptyHintreconstructed, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile75Ohm, 'empty 75Ohm', gfile75Ohm, high_cutoff_freq, 224754, 0, 0, 0, 0, 8)
#phasezeroedMintreconstructed, phasezeroedHintreconstructed, phasezeroedMoverH, phasezeroedpMminuspH, phasezeroedpMminuspHg, phasezeroedMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile75Ohm, 'empty 75Ohm', gfile75Ohm, high_cutoff_freq, 224754, 0, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#zeroedMintreconstructed, zeroedHintreconstructed, zeroedMoverH, zeroedpMminuspH, zeroedpMminuspHg, zeroedMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile75Ohm, 'empty 75Ohm', gfile75Ohm, high_cutoff_freq, 224754, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#standardMintreconstructed, standardHintreconstructed, standardMoverH, standardpMminuspH, standardpMminuspHg, standardMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellstandardfile75Ohm, 'standard 75Ohm', gfile75Ohm, high_cutoff_freq, 224818, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#sample75OhmMintreconstructed, sample75OhmHintreconstructed, sampleMoverH, samplepMminuspH, samplepMminuspHg, sampleMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellsamplefile75Ohm, 'sample 75Ohm', gfile75Ohm, high_cutoff_freq, 224753, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)

#emptyMintreconstructed, emptyHintreconstructed, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile1MOhm, 'empty 1MOhm', gfile1MOhm, high_cutoff_freq, 224773, 0, 0, 0, 0, 8)
#phasezeroedMintreconstructed, phasezeroedHintreconstructed, phasezeroedMoverH, phasezeroedpMminuspH, phasezeroedpMminuspHg, phasezeroedMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile1MOhm, 'empty 1MOhm', gfile1MOhm, high_cutoff_freq, 224773, 0, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#zeroedMintreconstructed, zeroedHintreconstructed, zeroedMoverH, zeroedpMminuspH, zeroedpMminuspHg, zeroedMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellemptyfile1MOhm, 'empty 1MOhm', gfile1MOhm, high_cutoff_freq, 224773, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#standardMintreconstructed, standardHintreconstructed, standardMoverH, standardpMminuspH, standardpMminuspHg, standardMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellstandardfile1MOhm, 'standard 1MOhm', gfile1MOhm, high_cutoff_freq, 224794, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)
#sample1MOhmMintreconstructed, sample1MOhmHintreconstructed, sampleMoverH, samplepMminuspH, samplepMminuspHg, sampleMoverH0, kstep = MDO3SpiceanalyzeMH.fundmagphase(outdir, kstep, ambrellsamplefile1MOhm, 'sample 1MOhm', gfile1MOhm, high_cutoff_freq, 224762, emptyMoverH, emptypMminuspH, emptypMminuspHg, emptyMoverH0, 8)

'''
plt.figure(100)
#plt.style.use('classic')
ax = plt.axes()
#ax.set_color_cycle([plt.cm.cool(i) for i in numpy.linspace(0, 1, numfiles)])
plt.title('synomag')
plt.grid(True)
plt.xlabel('H (kA/m)')
plt.ylabel('M (kA/m) ')
plt.xticks(rotation=90)
for i in range(numfiles):
#    if (i % 5) == 0:
    colorval = plt.cm.cool(hmax[i]/hmaxmax)    
    plt.plot(synomagHintreconstructedlist[i], synomagMintreconstructedlist[i], color = colorval)
#plt.plot(sample75OhmHintreconstructed, sample75OhmMintreconstructed, 'g', label = 'sample 75Ohm')
#plt.plot(sample1MOhmHintreconstructed, sample1MOhmMintreconstructed, 'm', label = 'sample 1MOhm')
#plt.plot(standardHintreconstructed, standardMintreconstructed, 'g', label = r'standard M vs H 1MOhm')
#plt.plot(zeroedHintreconstructed, zeroedMintreconstructed, 'b', label = 'should be zero M vs H 1MOhm')
#plt.ylim(0, 2*pi)
plt.xlim(-60, 60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.text(21, -0.55, r'synomag')
plt.savefig(figoutfile)
plt.show()

plt.figure(101)
#plt.style.use('classic')
ax = plt.axes()
#ax.set_color_cycle([plt.cm.rainbow(i) for i in numpy.linspace(0, 1, numfiles)])
#plt.title('synomag')
plt.grid(True)
plt.xlabel('H (kA/m)')
plt.ylabel('M (kA/m) ')
plt.xticks(rotation=90)
for i in range(numfiles):
#    if (i % 5) == 0:
    colorval = plt.cm.rainbow(hmax[i]/hmaxmax)
    plt.plot(synomagHintreconstructedlist[i], synomagMintreconstructedlist[i], color = colorval)
#plt.plot(sample75OhmHintreconstructed, sample75OhmMintreconstructed, 'g', label = 'sample 75Ohm')
#plt.plot(sample1MOhmHintreconstructed, sample1MOhmMintreconstructed, 'm', label = 'sample 1MOhm')
#plt.plot(standardHintreconstructed, standardMintreconstructed, 'g', label = r'standard M vs H 1MOhm')
#plt.plot(zeroedHintreconstructed, zeroedMintreconstructed, 'b', label = 'should be zero M vs H 1MOhm')
#plt.ylim(0, 2*pi)
plt.xlim(-60, 60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#plt.text(21, -0.55, r'synomag')
plt.savefig(figoutfile2)
plt.savefig(tiffoutfile1)
plt.show()

for i in range(len(slp)):
    slp[i] = slp[i]/cFe
    
plt.figure(102)
plt.scatter(hmax, slp)
plt.xlabel('H_{max} (kA/m)')
plt.ylabel('SLP (W/g(Fe))')
plt.savefig(tiffoutfileslp)

#print(hmax)
#print(slp)

plt.show()
