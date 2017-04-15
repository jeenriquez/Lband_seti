#!/usr/bin/env python

filenames = open('/Users/jeenriquez/RESEARCH/SETI_BL/L_band/hits_logistics/all_dat.lst').read().splitlines()

for filename in filenames:

    lines_to_read = open('/Users/jeenriquez/RESEARCH/SETI_BL/L_band/all_hits/'+filename+'.gpuspec.0000.dat').readlines()

    if 'File ID:' in str(lines_to_read[3:]):
        print 'Found double:%s'%filename


# /mnt_bls1/datax3/collate/AGBT16A_999_202/spliced_blc0001020304050607_guppi_57533_25698_HIP81742_0019.gpuspec.0000.fil
# /mnt_blc23/datax2/collate/AGBT16A_999_212/spliced_blc0001020304050607_guppi_57541_58714_HIP10670_0012.gpuspec.0000.fil
# /mnt_blc25/datax2/collate/AGBT16A_999_231/spliced_blc0001020304050607_guppi_57571_76298_HIP38325_0037.gpuspec.0000.fil
# /mnt_blc22/datax2/collate/AGBT16B_999_02/spliced_blc0001020304050607_guppi_57605_79074_Hip58001_0019.gpuspec.0000.fil
# /mnt_bls3/datax/collate/AGBT16A_999_251/spliced_blc0001020304050607_guppi_57636_60706_HIP48954_0008.gpuspec.0000.fil
# /mnt_blc21/datax2/collate/AGBT16A_999_103/spliced_blc02030405_2bit_guppi_57470_03186_HIP27918_0006.gpuspec.0000.fil
# /mnt_blc21/datax2/collate/AGBT16A_999_104/spliced_blc02030405_2bit_guppi_57470_44443_HIP83389_OFF_0004.gpuspec.0000.fil
# /mnt_blc21/datax2/collate/AGBT16A_999_104/spliced_blc02030405_2bit_guppi_57470_45453_HIP83389_0007.gpuspec.0000.fil


