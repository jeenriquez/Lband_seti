#!/usr/bin/env python

import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os
import matplotlib.pylab as plt

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'


#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.
#------

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

    obs_runs = df['obs_run'].unique()
    a_list = []
    list_targets =''
    list_A_stars=''
    i = 0

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

            B_list = []

            for ff,a_time in enumerate(list_a_star_times):

                #This delta_t is to find the closes observation to the current (basically to find the OFF).
                df_tmp = df_single_run[(df_single_run['Time stamp of first sample (MJD)'] > float(a_time)-0.1) & (df_single_run['Time stamp of first sample (MJD)'] < float(a_time)+0.1)]
                df_tmp['delta_t_off'] = df_tmp['Time stamp of first sample (MJD)'].apply(lambda x: float(x) - float(a_time))

                #---------------------------
                #Finding the A-B star pairs.

                jj = df_tmp[df_tmp['delta_t_off']>=0]['delta_t_off'].idxmin()   #Find A star's index

                if ff < 1:
                    RA = df_tmp['Source RA (J2000)'][jj]
                    DEC = df_tmp['Source DEC (J2000)'][jj]
                    obs_date = df_tmp['Gregorian date (YYYY/MM/DD)'][jj]

                try:
                    ii = df_tmp[df_tmp['delta_t_off']>0.001]['delta_t_off'].idxmin()   #Find B star's index  #.001 = 1.44 min
                    b_name = df_tmp['Source Name'][ii]

                    if 'DIAG_PSR' in b_name:
                        pdb.set_trace()

                    if a_star == b_name:
                        print 'WARNING: Skiping (a=b). ', a_star
                        raise ValueError('WARNING: Skiping (a=b). ', a_star)

                except:
                    b_name = ''


                B_list.append(b_name)

            #a_star_file_name, b_star_file_name
            tmp_string = [a_star,RA,DEC,obs_date,'['+','.join(B_list)+']']
            list_targets += '  &  '.join(tmp_string)+'\ \ \n'
            i+=1

            kk+=1
            print '------   o   -------'

        list_A_stars+=a_star+'\n'


    #---------------------------
    #Save lists

    with open('L_band_target_pairs.lst','w') as file_list:
        file_list.write(list_targets)

    t1 = time.time()
    print 'Search time: %5.2f min' % ((t1-t0)/60.)





