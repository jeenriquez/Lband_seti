#!/usr/bin/env python

import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os
import matplotlib.pylab as plt
from blimpy import Filterbank
import numpy as np

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'


#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.
#------


def plot_waterfall(fil, f_start=None, f_stop=None, if_id=0, logged=True,cb=True,MJD_time=False, **kwargs):
    """ Plot waterfall of data

    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """
    plot_f, plot_data = fil.grab_data(f_start, f_stop, if_id)

#
#     # Make sure waterfall plot is under 4k*4k
#     dec_fac_x, dec_fac_y = 1, 1
#     if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
#         dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]
#
#     if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
#         dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]
#
# #        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
#

    if MJD_time:
        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])
    else:
        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        cmap='viridis',
        **kwargs
    )
    if cb:
        plt.colorbar()
    plt.xlabel("Frequency [MHz]")
    if MJD_time:
        plt.ylabel("Time [MJD]")
    else:
        plt.ylabel("Time [s]")



def make_waterfall_plots(filenames_list,f_start,f_stop,ion = False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''
    #filutil spliced_blc0001020304050607_guppi_57802_28029_HIP72944_0002.gpuspec.0000.fil -b 1681.407 -e 1681.409 -p w

    if ion:
        plt.ion()

    n_plots = len(filenames_list)
    plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))



    for i,filename in enumerate(filenames_list):
        print filename

        plt.subplot(n_plots,1,i+1)

        fil = Filterbank(filename, f_start=f_start, f_stop=f_stop)
        plot_waterfall(fil,f_start=f_start, f_stop=f_stop,**kwargs)

        plt.ylabel('Time [s]')
        plt.title('')

    #Some plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xticks(np.arange(f_start, f_stop, (f_stop-f_start)/4.))

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    plt.subplots_adjust(hspace=0,wspace=0)
    if not ion:
        plt.show()

    plt.savefig('Candidate_waterfall_plots.v1.png')

def make_table(filename,init=False):
    '''
    '''

    if init:
        columns = ['FileID','Source','MJD','RA','DEC','TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 'FreqStart', 'FreqEnd', 'CoarseChanNum', 'FullNumHitsInRange','status','Hit_ID','ON_in_range','RFI_in_range']
        df_data = pd.DataFrame(columns=columns)

    else:

        file_dat = open(filename)
        hits = file_dat.readlines()

        #Info from header
        FileID = hits[1].strip().split(':')[-1].strip()
        Source = hits[3].strip().split(':')[-1].strip()

        MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
        RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
        DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

        #Info from individual Hits
        all_hits = [hit.strip().split('\t') for hit in hits[9:]]

        TopHitNum = zip(*all_hits)[0]
        DriftRate = zip(*all_hits)[1]
        SNR = zip(*all_hits)[2]
        Freq = zip(*all_hits)[3]
        ChanIndx = zip(*all_hits)[5]
        FreqStart = zip(*all_hits)[6]
        FreqEnd = zip(*all_hits)[7]
        CoarseChanNum = zip(*all_hits)[10]
        FullNumHitsInRange = zip(*all_hits)[11]

        data = {'TopHitNum':TopHitNum,
                'DriftRate':[float(df) for df in DriftRate],
                'SNR':[float(ss) for ss in SNR],
                'Freq':[float(ff) for ff in Freq],
                'ChanIndx':ChanIndx,
                'FreqStart':FreqStart,
                'FreqEnd':FreqEnd,
                'CoarseChanNum':CoarseChanNum,
                'FullNumHitsInRange':FullNumHitsInRange
                }

        df_data = pd.DataFrame(data)
        df_data = df_data.apply(pd.to_numeric)

        #Adding header information.
        df_data['FileID'] = FileID
        df_data['Source'] = Source.upper()
        df_data['MJD'] = MJD
        df_data['RA'] = RA
        df_data['DEC'] = DEC

        #Adding extra columns.
        df_data['Hit_ID'] = ''
        df_data['status'] = ''
        df_data['ON_in_range'] = ''
        df_data['RFI_in_range'] = ''

    return df_data

def calc_freq_range(hit,delta_t=0,max_dr=True,follow=False):
    '''Calculates the range of frequencies given a freq and drift_rate.
    '''

    if max_dr:
        drift_rate = MAX_DRIFT_RATE
    else:
        drift_rate = hit['DriftRate']

    if follow:
        freq = hit['Freq'] + drift_rate*(delta_t)/1000.
        delta_t = 2*OBS_LENGHT  # NOTE: I'm doing twice the amount. To widen the range a bit. Still it is dependend on hit['DriftRate']
    else:
        freq = hit['Freq']
        delta_t = delta_t+OBS_LENGHT  #adding to the end of the obs.

    low_bound  = freq - abs(drift_rate)*delta_t/1000.
    high_bound = freq + abs(drift_rate)*delta_t/1000.

    return [low_bound,high_bound]

def follow_candidate(hit,A_table,get_count=True):
    ''' Follows hit to another obs, and finds if antithing there.
    '''

    freq_range = calc_freq_range(hit,delta_t=A_table['delta_t'].values[0],max_dr=False,follow=True)
    new_A_table = A_table[(A_table['Freq']>freq_range[0]) & (A_table['Freq']<freq_range[1])]

    if get_count:
        n_hits_in_range = len(new_A_table)

        #Avoiding cases where multiple hits in one obs, and none in the other.
        if n_hits_in_range:
            return 1
        else:
            return 0
    else:
        return new_A_table

def find_candidates(A_table_list,B_table):
    '''Rejects hits based on some logic.
    '''

#     A1_table = A_table_list[0]
#     A2_table = A_table_list[1]
#     A3_table = A_table_list[2]
    A_table = pd.concat(A_table_list)

    #Removing hits outside the edges: 1100-1900 MHz.
    A_table = A_table[((A_table['Freq']>1100.) & (A_table['Freq']<1900.))]

    #Removing hits within Notch Filter:  1.2-1.34 GHz
    A_table = A_table[~((A_table['Freq']>1200.) & (A_table['Freq']<1340.))]

    #Removing non-drift signals
#     A1nd0_table = A1_table[A1_table['DriftRate'] != 0.0]
#     A2nd0_table = A2_table[A2_table['DriftRate'] != 0.0]
#     A3nd0_table = A3_table[A3_table['DriftRate'] != 0.0]
    And0_table = A_table[A_table['DriftRate'] != 0.0]

    #Make the SNR>25 cut.
    As25_table = And0_table[And0_table['SNR']> 25.]

    # Finding RFI within a freq range.
    if len(As25_table) > 0:
        As25_table['RFI_in_range'] = As25_table.apply(lambda hit: len(B_table[((B_table['Freq'] > calc_freq_range(hit)[0]) & (B_table['Freq'] < calc_freq_range(hit)[1]))]),axis=1)
        AnB_table = As25_table[As25_table['RFI_in_range'] == 0]
    else:
        print 'NOTE: Found no candidates.'
        return As25_table

    #Find the ones that are present in all the 3 ON obs, and follow the drifted signal.
    if len(AnB_table) > 2:
        A1nB_table = AnB_table[AnB_table['status'] == 'A1_table']
        A2nB_table = AnB_table[AnB_table['status'] == 'A2_table']
        A3nB_table = AnB_table[AnB_table['status'] == 'A3_table']
        if len(A1nB_table) > 0  and len(A2nB_table) > 0 and len(A3nB_table) > 0:

            A1nB_table['ON_in_range'] = A1nB_table.apply(lambda hit: follow_candidate(hit,A2nB_table) + follow_candidate(hit,A3nB_table) ,axis=1)
            AA_table = A1nB_table[A1nB_table['ON_in_range'] == 2]
        else:
            print 'NOTE: Found no candidates.'
            return  make_table('',init=True)
    else:
        print 'NOTE: Found no candidates.'
        return make_table('',init=True)

    if len(AA_table) > 0:
        AAA_table_list = []

        for hit_index, hit in AA_table.iterrows():
            A1i_table = follow_candidate(hit,A1nB_table,get_count=False)
            A1i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
            A2i_table = follow_candidate(hit,A2nB_table,get_count=False)
            A2i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)
            A3i_table = follow_candidate(hit,A3nB_table,get_count=False)
            A3i_table['Hit_ID'] = hit['Source']+'_'+str(hit_index)

            AAA_table_list += [A1i_table, A2i_table, A3i_table]

        AAA_table = pd.concat(AAA_table_list)
        pdb.set_trace()
    else:
        print 'NOTE: Found no candidates.'
        return AA_table


    return AAA_table

def remomve_RFI_regions(AAA_candidates):
    ''' Removing regions with lots of known RFI.
    '''

    #Removing corrupt files.
#    AAA_HIP69357 = AAA_candidates[AAA_candidates['Source']=='HIP69357']   # Want to look at the waterfall plots. Very strange behavior.
#    AAA_candidates = AAA_candidates[AAA_candidates['Source']!='HIP69357']

    #Removing Hydrogen line
#    Hydrogen_candidates = AAA_candidates[((AAA_candidates['Freq'] > 1420.) & (AAA_candidates['Freq'] < 1421.))]   # Could plot RA-DEC, and see distribution (maybe galactic plane??).
#    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1420.) & (AAA_candidates['Freq'] < 1421.))]

    #Removing Iridium signal.
#    AAA_Iridium = AAA_candidates[~((AAA_candidates['Freq'] > 1626.) & (AAA_candidates['Freq'] < 1626.5))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1615.) & (AAA_candidates['Freq'] < 1626.5))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1610.) & (AAA_candidates['Freq'] < 1628))]   #http://www.cv.nrao.edu/vla/upgrade/node114.html

    #Removing GOES signal
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1675.8) & (AAA_candidates['Freq'] < 1676.2))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1685.) & (AAA_candidates['Freq'] < 1686.))]#https://goes.gsfc.nasa.gov/text/goestechnotes.html

    #Removing bright GPS regions. http://www.gps.gov/systems/gps/modernization/civilsignals/
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1376.) & (AAA_candidates['Freq'] < 1386.))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1570.) & (AAA_candidates['Freq'] < 1580.))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1595.) & (AAA_candidates['Freq'] < 1610.))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1163.5) & (AAA_candidates['Freq'] < 1188.5 ))]  #L5 #https://www.rfglobalnet.com/doc/1176-mhz-gps-l5-band-ceramic-notch-filter-0001
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1374.) & (AAA_candidates['Freq'] < 1376.))] #https://www.naic.edu/~phil/rfi/1375_rfi_nov09.html
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1544.) & (AAA_candidates['Freq'] < 1545.))]  #Galileo SAR Downlink


    #Mobile-satellite communications.
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1555.) & (AAA_candidates['Freq'] < 1559.))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1656.5) & (AAA_candidates['Freq'] < 1660.5))]
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1545.) & (AAA_candidates['Freq'] < 1547.))] #Alphasat 1545.5MHz + Inmarsat Aero
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1626.5) & (AAA_candidates['Freq'] < 1660.5))]  #TowardsthePersonalCommunicationsEnvironment

    #removing from    1520-1560?

    #http://www.infineon.com/dgdl/Infineon-AN420_BGA524N6-AN-v01_00-EN.pdf?fileId=5546d462518ffd8501520ce7a08570fc
    #http://www.navipedia.net/index.php/File:GNSS_All_Signals.png

    #NOAA 17,18	1707 MHz	1700-1710 MHz	Meteorological-satellite service
    AAA_candidates = AAA_candidates[~((AAA_candidates['Freq'] > 1700.) & (AAA_candidates['Freq'] < 1710.))]

    return AAA_candidates

if __name__ == "__main__":

    #---------------------------
    #Setting things up.

    project_dir = '/Users/jeenriquez/RESEARCH/SETI_BL/L_band/'

    t0 = time.time()

    #---------------------------
    # Read in the full "A list" of stars.
    #---------------------------
    a_list_master = open(project_dir+'A_stars.lst').read().splitlines()

    all_dats = 'Need to open here the dats file.'

    #---------------------------
    # Find the good  data by:
    # - Reading in the output from spider.
    #---------------------------
    # Initial data frame set up

    spider_filename = project_dir+'/spider_outs/spider2.out.04.11.17'

    try:
        df = pd.read_csv(spider_filename, sep=",|=", header=None,engine='python')
    except:
        IOError('Error opening file: %s'%filename)

    df2 = df.ix[:,1::2]
    df2.columns = list(df.ix[0,0::2])


    #Selection of high resolution data
    df = df2[df2['file'].str.contains("gpuspec.0000.fil",na=False)]

    #Selection of observations from given band (soft)
    df = df[df['Frequency of channel 1 (MHz)'] < 2500]

    #Selection of observations from blc30 (since also in bls1).
    df = df[~df['file'].str.contains('mnt_blc30')]

    #---------------------------
    #Check for correct Number of Samples
    df = df[df['Number of samples'] == 16 ]

    #---------------------------
    # Adding some extra columns for later look at the good set of data.

    df['bands_used'] = [df['file'][ii].split('/')[-1].split('_')[1].replace('blc','') for ii in df.index]
    df['mid_Freq'] = df['Frequency of channel 1 (MHz)']-2.835503418452676e-06*df['Number of channels']/2.
    df['mid_Freq2'] = df['Frequency of channel 1 (MHz)']-2.7939677238464355e-06*df['Number of channels']/2.

    df['Source Name'] = df['Source Name'].str.upper()

    #---------------------------
    #Check the data that has the 4 good bands (02030405).
    df = df[df['bands_used'].str.contains('02030405')]

    #---------------------------
    #Check for high resolution data with the bad central frequency.
    # The two main frequency resolutions used are -2.7939677238464355e-06 and -2.835503418452676e-06  .

    df_good_mid_Freq = df[((df['mid_Freq'] > 1501.4) & (df['mid_Freq'] < 1501.5))]
    df_good_mid_Freq2 = df[((df['mid_Freq2'] > 1501.4) & (df['mid_Freq2'] < 1501.5))]

    df = pd.concat([df_good_mid_Freq,df_good_mid_Freq2])

    #---------------------------
    # Apply format change in MJD
    df['Time stamp of first sample (MJD)'] = df['Time stamp of first sample (MJD)'].apply(pd.to_numeric)

    #---------------------------
    #Adding extra columns

    #df['obs_run'] = df.apply(lambda x: x['file'].split('collate/')[-1].split('/spliced')[0])
    df['obs_run'] = [df['file'][ii].split('collate/')[-1].split('/spliced')[0] for ii in df.index]
    #df['filename_dat'] = df.apply(lambda x: 'spliced'+x['file'].split('/spliced')[-1].replace('.fil','.h5'))
    df['filename_dat'] = ['spliced'+df['file'][ii].split('/spliced')[-1].replace('.fil','.dat') for ii in df.index]

    # #---------------------------
    # # Selecting only the targets in the A-list
    #
    # #Selecting all the targets from the B list
    # df_targets_blist = df[~df['Source Name'].isin(a_list_master)]
    # df_targets_clist =  df[df['Source Name'].str.contains('_OFF',na=False)]
    # df_targets_blist = pd.concat([df_targets_blist,df_targets_clist])
    # else_list = df_targets_blist['Source Name'].unique()
    #
    # #Selecting all the good targets from the A list
    # df_targets_alist = df[~df['Source Name'].isin(else_list)]
    #
    # #---------------------------
    # #Showing some info
    #
    # print '------      o      --------'
    # a_unique = df_targets_alist['Source Name'].unique()
    # print 'This list was created for the L band data'
    # print 'The total number of targets from the A-list that:'
    # print 'Observed and spliced is      : %i'%(len(a_unique))
    #
    # #---------------------------
    # #Group the df_targets and look for the ones observed 3 times or more
    # # Grouping without date constrains.
    # df_targets = df_targets_alist.groupby('Source Name').count()['file'] > 2
    # df_bool_list = df_targets.tolist()
    # list_completed = list(df_targets[df_bool_list].index.values)
    #
    # #---------------------------
    # #Selecting targets with "completed" observations
    # df_targets_alist = df_targets_alist[df_targets_alist['Source Name'].isin(list_completed)]
    # alist_completed_unique = df_targets_alist['Source Name'].unique()
    #
    # print 'Have at least 3 observations : %i'%(len(alist_completed_unique))




    #---------------------------
    #Looping over observing run.

    AAA_candidates_list = []
    all_candidates_list = []
    AAA_candidates = make_table('',init=True)
    obs_runs = df['obs_run'].unique()
    a_list = []

    for obs_run in obs_runs:

        df_single_run =  df[df['obs_run'] == obs_run]

        #---------------------------
        # Selecting only the targets in the A-list

        #Selecting all the targets from the B list
        df_targets_blist = df_single_run[~df_single_run['Source Name'].isin(a_list_master)]
        df_targets_clist =  df_single_run[df_single_run['Source Name'].str.contains('_OFF',na=False)]
        df_targets_blist = pd.concat([df_targets_blist,df_targets_clist])
        else_list = df_targets_blist['Source Name'].unique()

        #Selecting all the good targets from the A list
        df_targets_alist = df_single_run[~df_single_run['Source Name'].isin(else_list)]

        #---------------------------
        #Group the df_targets and look for the ones observed 3 times or more
        df_targets = df_targets_alist.groupby('Source Name').count()['file'] > 2
        df_bool_list = df_targets.tolist()
        list_completed = list(df_targets[df_bool_list].index.values)

        #---------------------------
        #Selecting targets with a complete set of observations
        df_targets_alist = df_targets_alist[df_targets_alist['Source Name'].isin(list_completed)]
        alist_completed_unique = list(df_targets_alist['Source Name'].unique())

        a_list += alist_completed_unique
        #---------------------------
        #Looping over the A stars

        for a_star in alist_completed_unique:
            kk = 1

            df_a_star = df_targets_alist[df_targets_alist['Source Name'] == a_star]
            list_a_star_times = df_a_star['Time stamp of first sample (MJD)'].unique()

            #---------------------------
            # Reading hits data for A-B pairs.
            print 'Reading hits data for %s.'%a_star

    #         A_table = make_table('',init=True)
    #         B_table = make_table('',init=True)
            A_table_list = []
            B_table_list = []

            for a_time in list_a_star_times:

                #This delta_t is to find the closes observation to the current (basically to find the OFF).
                df_tmp = df_single_run[(df_single_run['Time stamp of first sample (MJD)'] > float(a_time)-0.1) & (df_single_run['Time stamp of first sample (MJD)'] < float(a_time)+0.1)]
                df_tmp['delta_t_off'] = df_tmp['Time stamp of first sample (MJD)'].apply(lambda x: float(x) - float(a_time))

                #---------------------------
                #Finding the A-B star pairs.

                jj = df_tmp[df_tmp['delta_t_off']>=0]['delta_t_off'].idxmin()   #Find A star's index

                if not os.path.isfile(project_dir+'all_hits/'+df_tmp['filename_dat'][jj]):
                    print 'WARNING: missing ON file: %s'%(str(df_tmp['filename_dat'][jj]))

                Ai_table=make_table(project_dir+'all_hits/'+df_tmp['filename_dat'][jj])
                Ai_table['status'] = 'A%i_table'%kk
                #Info from df.
                Ai_table['filename_fil'] = df_tmp['file'][jj]
                Ai_table['obs_run'] = df_tmp['obs_run'][jj]
                Ai_table['delta_t'] = (a_time-list_a_star_times[0])*3600*24  # In sec

                try:
                    ii = df_tmp[df_tmp['delta_t_off']>0.001]['delta_t_off'].idxmin()   #Find B star's index  #.001 = 1.44 min

                    b_name = df_tmp['Source Name'][ii]

                    if a_star == b_name:
                        print 'WARNING: Skiping (a=b). ', a_star
                        raise ValueError('WARNING: Skiping (a=b). ', a_star)

                    if not os.path.isfile(project_dir+'all_hits/'+df_tmp['filename_dat'][ii]):
                        print 'WARNING: missing OFF file: %s'%(str(df_tmp['filename_dat'][ii]))

                    Bi_table=make_table(project_dir+'all_hits/'+df_tmp['filename_dat'][ii])
                    Bi_table['status'] = 'B%i_table'%kk

                    #Info from df.
                    Bi_table['filename_fil'] = df_tmp['file'][ii]
                    Bi_table['obs_run'] = df_tmp['obs_run'][ii]

                except:
                    Bi_table=make_table('',init=True)

                #---------------------------
                #Grouping all hits per obs set.
                A_table_list.append(Ai_table)
                B_table_list.append(Bi_table)

                kk+=1

            #Concatenating
            A_table = pd.concat(A_table_list)
            B_table = pd.concat(B_table_list)

            #To save all the hits. Uncomment these 3 lines.
#             all_candidates_list.append(A_table)  This blows up the mem. Caution.
#             all_candidates_list.append(B_table)
#            continue

            print 'Finding all candidates for this A-B set.'
#             AAA_table = find_candidates(A_table_list[0],A_table,B_table)
            AAA_table = find_candidates(A_table_list,B_table)

            if len(AAA_table) > 0:
                print 'Found: %2.2f'%(len(AAA_table)/3.)

#             if a_star == 'HIP33955':
#                 pdb.set_trace()

            AAA_candidates_list.append(AAA_table)
            print '------   o   -------'



    t1 = time.time()
    print 'Search time: %.2f min' % ((t1-t0)/60.)

    stop

    #Concatenating all the candidates.
    AAA_candidates = pd.concat(AAA_candidates_list,ignore_index=True)
    AAA_candidates_list = 0.

    #Save hits.
    AAA_candidates.to_csv('AAA_candidates.v4_%.0f.csv'%time.time())

    #Looking at some stats.
    plt.ion()
    plt.figure()
    AAA_candidates['Freq'].plot.hist(bins=100,logy=True)

    plt.figure()
    AAA_candidates['DriftRate'].plot.hist(bins=50)

    stop



    #Removing a bunch of RFI regions (GPS and so on).
    AAA_candidates = remomve_RFI_regions(AAA_candidates)

