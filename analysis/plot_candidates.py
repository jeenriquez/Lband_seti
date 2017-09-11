#!/usr/bin/env python

import pandas as pd
import socket
import pdb;# pdb.set_trace()
import time
import os
import matplotlib.pylab as plt
from blimpy import Filterbank
from blimpy import Waterfall
import numpy as np
from blimpy.utils import db, lin, rebin, closest
import find_candidates

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.

MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot
MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot
MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory
MAX_HEADER_BLOCKS   = 100                    # Max size of header (in 512-byte blocks)

#------


def plot_waterfall(fil, f_start=None, f_stop=None, if_id=0, logged=True,cb=False,MJD_time=False, **kwargs):
    """ Plot waterfall of data

    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """

    fontsize=15

    plot_f, plot_data = fil.grab_data(f_start, f_stop, if_id)


    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]

    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]

    plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)

    if MJD_time:
        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])
    else:
        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    this_plot = plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        cmap='viridis_r',
        **kwargs
    )
    if cb:
        plt.colorbar()
    plt.xlabel("Frequency [MHz]",fontsize=fontsize)
    if MJD_time:
        plt.ylabel("Time [MJD]",fontsize=fontsize)
    else:
        plt.ylabel("Time [s]",fontsize=fontsize)

    return this_plot

def make_waterfall_plots(filenames_list,target,f_start,f_stop,ion = False,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''
    #filutil spliced_blc0001020304050607_guppi_57802_28029_HIP72944_0002.gpuspec.0000.fil -b 1681.407 -e 1681.409 -p w

    fontsize=15

    if ion:
        plt.ion()

    if target == 'HIP45493':
#         f_start = 1528.4
#         f_stop  = 1528.49
        xf_start = 1528.41
        xf_stop  = 1528.483
        y_start = 0.004
        yf_stop  = 0.004

    if target == 'HIP7981':
#         f_start -= 0.019
#         f_stop  += 0.019
        f_start -= 0.3
        f_stop  += 0.3
    else:
      return None
#      print target

    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    fil = Filterbank(filenames_list[0], f_start=f_start, f_stop=f_stop)
    A1_avg = fil.data.mean()
    A1_max = fil.data.max()
    A1_std = np.std(fil.data)

    labeling = ['A','B','A','C','A','D']

    for i,filename in enumerate(filenames_list):
        print filename
        plt.subplot(n_plots,1,i+1)

        fil = Filterbank(filename, f_start=f_start, f_stop=f_stop)
#        this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg,vmax=A1_avg+10.*A1_std,**kwargs)
        this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg-A1_std*0,vmax=A1_avg+10.*A1_std,**kwargs)

#        plt.ylabel('Time [s]',fontsize=fontsize)
#        fig[1][i].text(f_start + 4.5*(f_stop-f_start)/5., 50 ,labeling[i],color='w',fontsize=fontsize)

        if i == 0:
            plt.title(target)
            cax = fig[0].add_axes([0.9, 0.1, 0.03, 0.8])
            fig[0].colorbar(this_plot,cax=cax,label='Power')

    #Some plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xticks(np.arange(f_start, f_stop, (f_stop-f_start)/4.))

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    plt.subplots_adjust(hspace=0,wspace=0)
    if not ion:
        plt.show()

    plt.savefig('Candidate_waterfall_plots.'+target+'.png')
#    plt.savefig('Candidate_waterfall_plots.'+target+'.eps', format='eps', dpi=300)

def get_filenames_list(target):

    if target =='HIP17147':
        filenames_list=['/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_69379_HIP17147_0015.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_69728_HIP16229_0016.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_70077_HIP17147_0017.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_70423_HIP16299_0018.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_70769_HIP17147_0019.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_192/spliced_blc0001020304050607_guppi_57523_71116_HIP16341_0020.gpuspec.0000.fil']

    elif target =='HIP20901':
        filenames_list=['/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_50058_Hip20901_0027.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_50409_Hip19822_0028.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_50760_Hip20901_0029.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_51107_Hip19834_0030.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_51453_Hip20901_0031.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16B_999_03/spliced_blc0001020304050607_guppi_57606_51802_Hip19862_0032.gpuspec.0000.fil']

    elif target =='HIP39826':
        filenames_list=['/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_02669_HIP39826_0009.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_03002_HIP39826_OFF_0010.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_03335_HIP39826_0011.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_03668_HIP39826_OFF_0012.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_04001_HIP39826_0013.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_79/spliced_blc02030405_2bit_guppi_57456_04334_HIP39826_OFF_0014.gpuspec.0000.fil']

    elif target =='HIP4436':
        filenames_list=['/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_80733_HIP4436_0032.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_81079_HIP3333_0033.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_81424_HIP4436_0034.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_81768_HIP3597_0035.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_82111_HIP4436_0036.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT17A_999_08/spliced_blc0001020304050607_guppi_57803_82459_HIP3677_0037.gpuspec.0000.fil']

    elif target =='HIP45493':
        filenames_list=['/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_55512_Hip45493_0045.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_55850_Hip44654_0046.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_56188_Hip45493_0047.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_56535_Hip44877_0048.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_56882_Hip45493_0049.gpuspec.0000.fil',
        '/mnt_blc24/datax2/collate/AGBT16A_999_249/spliced_blc0001020304050607_guppi_57599_57219_Hip45377_0050.gpuspec.0000.fil']

    elif target =='HIP65352':
        filenames_list=['/mnt_blc21/datax2/collate/AGBT16A_999_84/spliced_blc02030405_2bit_guppi_57459_34297_HIP65352_0027.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_84/spliced_blc02030405_2bit_guppi_57459_34623_HIP65352_OFF_0028.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_84/spliced_blc02030405_2bit_guppi_57459_34949_HIP65352_0029.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_84/spliced_blc02030405_2bit_guppi_57459_35275_HIP65352_OFF_0030.gpuspec.0000.fil',
        '/mnt_blc21/datax2/collate/AGBT16A_999_84/spliced_blc02030405_2bit_guppi_57459_35601_HIP65352_0031.gpuspec.0000.fil']

    elif target =='HIP66704':
        filenames_list=['/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_54573_Hip66704_0003.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_54910_Hip65678_0004.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_55247_Hip66704_0005.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_55588_Hip65946_0006.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_55929_Hip66704_0007.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_25/spliced_blc0001020304050607_guppi_57650_56272_Hip66192_0008.gpuspec.0000.fil']

    elif target =='HIP74981':
        filenames_list=['/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_22406_HIP74981_0003.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_22755_HIP74284_0004.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_23103_HIP74981_0005.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_23457_HIP74315_0006.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_23810_HIP74981_0007.gpuspec.0000.fil',
        '/mnt_blc22/datax2/collate/AGBT16A_999_191/spliced_blc0001020304050607_guppi_57523_24142_HIP74439_0008.gpuspec.0000.fil']

    elif target =='HIP7981':
        filenames_list=['/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_15520_HIP7981_0039.gpuspec.0000.fil',
        '/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_15867_HIP6917_0040.gpuspec.0000.fil',
        '/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_16219_HIP7981_0041.gpuspec.0000.fil',
        '/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_16566_HIP6966_0042.gpuspec.0000.fil',
        '/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_16915_HIP7981_0043.gpuspec.0000.fil',
        '/mnt_bls3/datax2/collate/AGBT16B_999_48/spliced_blc0001020304050607_guppi_57680_17264_HIP6975_0044.gpuspec.0000.fil']

    elif target =='HIP82860':
        filenames_list=['/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_79761_HIP82860_0027.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_80108_HIP81813_0028.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_80455_HIP82860_0029.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_80802_HIP82056_0030.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_81149_HIP82860_0031.gpuspec.0000.fil',
        '/mnt_bls3/datax/collate/AGBT16B_999_35/spliced_blc0001020304050607_guppi_57664_81493_HIP82518_0032.gpuspec.0000.fil']

    elif target =='HIP99427':
        filenames_list=['/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_83026_HIP99427_0033.gpuspec.0000.fil',
        '/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_83371_HIP100670_0034.gpuspec.0000.fil',
        '/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_83716_HIP99427_0035.gpuspec.0000.fil',
        '/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_84066_HIP99560_0036.gpuspec.0000.fil',
        '/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_84416_HIP99427_0037.gpuspec.0000.fil',
        '/mnt_blc20/datax2/collate/AGBT16B_999_105/spliced_blc0001020304050607_guppi_57752_84766_HIP99759_0038.gpuspec.0000.fil']
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57754_84892_HIP99427_0036.gpuspec.0000.fil',
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57754_85226_HIP100670_0037.gpuspec.0000.fil',
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57754_85560_HIP99427_0038.gpuspec.0000.fil',
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57754_85900_HIP99560_0039.gpuspec.0000.fil',
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57754_86240_HIP99427_0040.gpuspec.0000.fil',
#         '/mnt_blc20/datax2/collate/AGBT16B_999_108/spliced_blc0001020304050607_guppi_57755_00180_HIP99759_0041.gpuspec.0000.fil',

    else:
        raise ValueError('No known target',target)

    return filenames_list


def get_data(filenames_list,target,dat_dit,f_start,f_stop):
    ''' save some data.
    '''

    for i,filename in enumerate(filenames_list):

        fil_file = Filterbank(filename, f_start=f_start, f_stop=f_stop)
        new_filename = filename.replace('.fil','.h5').split('/')[-1]
        fil_file.write_to_hdf5(dat_dit+new_filename)


if __name__ == "__main__":
    ''' Make it happen moment.
    '''

    #---------------------------
    # Read in the full "A list" of stars
    # This comes from the BL database.
    #---------------------------
    local_host = socket.gethostname()

    if 'bl' in local_host:
        dat_dit = '/datax/users/eenriquez/L_band_headquarters/hits_logistics/'

    else:
        dat_dit = '/Users/jeenriquez/RESEARCH/software/Lband_seti/analysis/'

    #---------------------------
    AAA_candidates = pd.read_csv(dat_dit+'AAA_candidates.v4_1492476400.csv')
    targets = list(AAA_candidates.groupby('Source').count().index)

    table_events =''

    for target in targets:
        AAA_single = AAA_candidates[AAA_candidates['Source'] == target]
        print target
        filenames_list = get_filenames_list(target)

        AAA1_single = AAA_single[AAA_single['status'] == 'A1_table'].sort('SNR')

        f_start = AAA1_single['Freq'].values[-1] - 0.001
        f_stop = AAA1_single['Freq'].values[-1] + 0.001
        coarse_channel=AAA1_single['CoarseChanNum'].values[-1]

        make_waterfall_plots(filenames_list,target,f_start,f_stop,ion=True)

#        get_data(filenames_list,target,dat_dit,f_start-0.099,f_stop+0.099)


        for_table = [AAA1_single['Source'].values[0],'%.5f'%AAA1_single['Freq'].values[-1],'%.3f'%AAA1_single['DriftRate'].values[-1],'%.1f'%AAA1_single['SNR'].values[-1]]
        table_events+='  &  '.join(for_table)+'\ \ \n'

    #Making table of events
    with open('L_band_top_events.lst','w') as file_list:
        file_list.write(table_events)

    stop
    #Removing a bunch of RFI regions (GPS and so on).
    AAA_candidates = remomve_RFI_regions(AAA_candidates)


