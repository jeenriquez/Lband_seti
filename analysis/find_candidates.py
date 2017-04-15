#!/usr/bin/env python


import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os
import matplotlib.pylab as plt

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

def make_table(filename,init=False):
    '''
    '''

    if init:
        columns = ['FileID','Source','MJD','RA','DEC','TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 'FreqStart', 'FreqEnd', 'CoarseChanNum', 'FullNumHitsInRange','status','Hit_ID']
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
        df_data['Source'] = Source
        df_data['MJD'] = MJD
        df_data['RA'] = RA
        df_data['DEC'] = DEC

        #Adding extra columns.
        df_data['Hit_ID'] = ''
        df_data['status'] = ''

    return df_data

def find_candidates(A1_table,A_table,B_table):
    '''Rejects hits based on some logic.
    '''

    #Removing hits outside the edges: 1100-1900 MHz.
    A1_table = A1_table[((A1_table['Freq']>1100.) & (A1_table['Freq']<1900.))]

    #Removing hits within Notch Filter:  1.2-1.34 GHz
    A1_table = A1_table[~((A1_table['Freq']>1200.) & (A1_table['Freq']<1340.))]

    #Removing non-drift signals
    And0_table = A1_table[A1_table['DriftRate'] != 0.0]

    #Make the SNR>25 cut.
    As25_table = And0_table[And0_table['SNR']> 25.]

    ## Removing hits that show at the exact Freq than in B.
    # AA_table = As25_table[~As25_table['Freq'].isin(B_table['Freq'])]

    if len(As25_table) > 0:
        # Finding RFI within a freq range.
        As25_table['RFI_in_range'] = As25_table.apply(lambda x: len(B_table[((B_table['Freq']>x['FreqStart']) & (B_table['Freq']<x['FreqEnd']))]),axis=1)
        AnB_table = As25_table[As25_table['RFI_in_range'] == 0]
    else:
        print 'NOTE: Found no candidates.'
        return As25_table

    if len(AnB_table) > 0:
        #Find the ones that are present in all the 3 ON obs.
        AnB_table['ON_in_range'] = AnB_table.apply(lambda x: len(A_table[((A_table['Freq']>x['FreqStart']) & (A_table['Freq']<x['FreqEnd']))]),axis=1)
        AA_table = AnB_table[AnB_table['ON_in_range'] > 2]
    else:
        print 'NOTE: Found no candidates.'
        return AnB_table

    if len(AA_table) > 0:
        AAA_table_list = []

        for row_index, row in AA_table.iterrows():
            Ai_table = A_table[((A_table['Freq']>row['FreqStart']) & (A_table['Freq']<row['FreqEnd']))]
            Ai_table['Hit_ID'] = row['Source']+str(row_index)
            AAA_table_list.append(Ai_table)
        AAA_table = pd.concat(AAA_table_list)
    else:
        print 'NOTE: Found no candidates.'
        return AA_table


    return AAA_table



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

#
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
AAA_candidates = make_table('',init=True)
obs_runs = df['obs_run'].unique()
a_list = []

for obs_run in obs_runs:

    df_single_run = df[df['obs_run'].str.contains(obs_run,na=False)]

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

            df_tmp = df_single_run[(df_single_run['Time stamp of first sample (MJD)'] > float(a_time)-0.1) & (df_single_run['Time stamp of first sample (MJD)'] < float(a_time)+0.1)]
            df_tmp['delta_t'] = df_tmp['Time stamp of first sample (MJD)'].apply(lambda x: float(x) - float(a_time))

            #---------------------------
            #Finding the A-B star pairs.

            jj = df_tmp[df_tmp['delta_t']>=0]['delta_t'].idxmin()   #Find A star index

            if not os.path.isfile(project_dir+'all_hits/'+df_tmp['filename_dat'][jj]):
                print 'WARNING: missing file: %s'%(str(df_tmp['filename_dat'][jj]))
                continue

            Ai_table=make_table(project_dir+'all_hits/'+df_tmp['filename_dat'][jj])
            Ai_table['status'] = 'A%i_table'%kk

            try:
                ii = df_tmp[df_tmp['delta_t']>0.001]['delta_t'].idxmin()   #Find B star index  #.001 = 1.44 min

                if not os.path.isfile(project_dir+'all_hits/'+df_tmp['filename_dat'][ii]):
                    print 'WARNING: missing file: %s'%(str(df_tmp['filename_dat'][ii]))

                Bi_table=make_table(project_dir+'all_hits/'+df_tmp['filename_dat'][ii])
                Bi_table['status'] = 'B%i_table'%kk

            except:
                Bi_table=make_table('',init=True)

#             if kk == 1:
#                 A1_table=make_table(project_dir+'all_hits/'+df_tmp['filename_dat'][jj])
#                 A1_table['status'] = 'A1_table'

            #---------------------------
            #Grouping all hits per obs set.
#             A_table = pd.concat([A_table,Ai_table])
#             B_table = pd.concat([B_table,Bi_table])
            A_table_list.append(Ai_table)
            B_table_list.append(Bi_table)

            kk+=1

        #Concatenating
        A_table = pd.concat(A_table_list)
        B_table = pd.concat(B_table_list)

        print 'Finding all candidates for this A-B set.'
        AAA_table = find_candidates(A_table_list[0],A_table,B_table)

        if len(AAA_table) > 0:
            print 'Found: %2.2f'%(len(AAA_table)/3.)

        AAA_candidates_list.append(AAA_table)
#        print 'Now the total is: %2.2f'%(sum([len(aa) for aa in AAA_candidates_list])/3.)
        print '------   o   -------'
        stop

#Concatenating all the candidates.
AAA_candidates = pd.concat(AAA_candidates_list,ignore_index=True)

t1 = time.time()
print 'Search time: %5.2f min' % ((t1-t0)/60.)


stop


#Looking at some stats.
plt.ion()
plt.figure()
AAA_candidates['Freq'].plot.hist(bins=100,logy=True)

plt.figure()
AAA_candidates['DriftRate'].plot.hist(bins=50)




#filutil spliced_blc0001020304050607_guppi_57802_28029_HIP72944_0002.gpuspec.0000.fil -b 1681.407 -e 1681.409 -p w
