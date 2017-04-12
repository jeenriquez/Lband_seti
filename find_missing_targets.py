#!/usr/bin/env python

import socket
local_host = socket.gethostname()

#---------------

Lband_analysis_dir = '/datax/users/eenriquez/L-band_analysis/'

targets = open(Lband_analysis_dir+'L_band_target_pairs.lst').readlines()

try1_targets = open('/home/eenriquez/software/Lband_seti/data_logs/turbo_seti_done_try1_all.lst').readlines()
try2_targets = open('/home/eenriquez/software/Lband_seti/data_logs/turbo_seti_done_try2_all.lst').readlines()


#---------------
#How to get the missing targets which should be processed
missing_targets = [target for target in targets if target.split('spliced')[-1].split('.gpu')[0] not in str(try1_targets)+str(try2_targets)]


#---------------
#Saving file

with open('/datax/users/eenriquez/L-band_analysis/%s_missing_targets.lst'%local_host,'w') as mt_file:
    mt_file.write('\n'.join(missing_targets))
