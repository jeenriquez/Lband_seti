#!/usr/bin/env python

import pandas as pd
import matplotlib
import matplotlib.pylab as plt

#------------------------------------------------
#setting up

dir_loc = '/datax/users/eenriquez/L_band_headquarters/hits_logistics/'

print 'Reading data.'
AAA_candidates = pd.read_csv(dir_loc+'All_hits_turbo_seti.csv')
AAA_candidates_V4 = pd.read_csv(dir_loc+'AAA_candidates.v4_1492476400.csv')


#------------------------------------------------
print 'Making first cut.'

#Removing hits outside the edges: 1100-1900 MHz.
AAA_candidates = AAA_candidates[((AAA_candidates['Freq']>1100.) & (AAA_candidates['Freq']<1900.))]

#Removing hits within Notch Filter:  1.2-1.34 GHz
AAA_candidates = AAA_candidates[~((AAA_candidates['Freq']>1200.) & (AAA_candidates['Freq']<1340.))]

#Removing non-drift signals
AAA_candidates_cut = AAA_candidates[AAA_candidates['DriftRate'] != 0.0]

#Make the SNR>25 cut.
AAA_candidates_cut = AAA_candidates_cut[AAA_candidates_cut['SNR']> 25.]

#------------------------------------------------
#Plotting

fontsize=18
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : fontsize}

matplotlib.rc('font', **font)
plt.ion()
plt.figure(figsize=[10,8])
print 'Plotting Frequency Histrogram.'
plt.clf()
AAA_candidates['Freq'].plot.hist(bins=200,logy=True,color='#7B90C7',edgecolor = "none",label='All hits')
AAA_candidates_cut['Freq'].plot.hist(bins=200,logy=True,color='#3E4F8A',edgecolor = "none",label='No-drift and SNR cut')
AAA_candidates_V4['Freq'].plot.hist(bins=50,logy=True,color='orange',edgecolor = "none",label='Most Significant hits')

plt.xlabel('Frequency [MHz]',fontdict=font)
plt.ylabel('Counts [log]',fontdict=font)
plt.legend()
plt.savefig('Frequency_hist.png')
plt.savefig('Frequency_hist.pdf', format='pdf', dpi=400)

plt.figure(figsize=[10,8])
print 'Plotting Drift Rate Histrogram.'
plt.clf()
AAA_candidates['DriftRate'].plot.hist(bins=51,logy=True,color='#7B90C7',edgecolor = "none",label='All hits')
AAA_candidates_cut['DriftRate'].plot.hist(bins=51,logy=True,color='#3E4F8A',edgecolor = "none",label='No-drift and SNR cut')
AAA_candidates_V4['DriftRate'].plot.hist(bins=51,logy=True,color='orange',edgecolor = "none",label='Most Significant hits')

plt.xlabel('Drift Rate [Hz/sec]',fontdict=font)
plt.ylabel('Counts [log]',fontdict=font)
plt.legend()
plt.savefig('Drift_hist.png')
plt.savefig('Drift_hist.pdf', format='pdf', dpi=400)
