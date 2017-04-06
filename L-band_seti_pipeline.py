#!/usr/bin/env python

from argparse import ArgumentParser
import socket
import subprocess
from blimpy import Waterfall
from optparse import OptionParser
import sys

def reset_outs():
    '''
    '''

    out = None
    err = None

    return out,err

p = OptionParser()
p.set_usage('python L-band_seti_pipeline.py <FULL_PATH_TO_FIL_FILE> [options]')
p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='./', help='Location for output files. Default: local dir. ')
p.add_option('-s', '--star', dest='star', type='str', default='./', help='Location for output files. Default: local dir. ')
opts, args = p.parse_args(sys.argv[1:])

working_dir = './'

star = opts.star
out_dir = opts.out_dir

#------------------------------------

star_path = star.split('spliced')[0]
star_name = 'spliced'+star.replace('.fil','.h5').split('spliced')[-1]

#------------------------------------
#Run turbo_seti

out,err = reset_outs()

command=['python','/home/eenriquez/software/bl-soft/turbo_seti/bin/seti_event.py',star,'-M','2']
print ' '.join(command)
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(out, err) = proc.communicate()

with open(out_dir+star_name.replace('.h5','.seti_event.log'), 'a') as f:
    f.write(out)

if err or proc.returncode != 0:
    with open(out_dir+star_name.replace('.h5','.seti_event.err'), 'a') as f:
        f.write(err)


