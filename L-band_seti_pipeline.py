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

local_host = socket.gethostname()


def cmd_tool(args=None):
    """ Command line tool for plotting and viewing info on filterbank files """

    p = OptionParser()
    p.set_usage('python L-band_seti_pipeline.py <FULL_PATH_TO_FIL_FILE> [options]')
    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='L-band_out', help='Location for output files. Default: local dir. ')
    p.add_option('-n', '--node', dest='node', type='str', default=local_host, help='Name of host node.')
    opts, args = p.parse_args(sys.argv[1:])

    out_dir = opts.out_dir
    node = opts.node

    #------------------------------------
    #Check list of files todo

    todo_list = '/mnt_'+node+'/datax/users/eenriquez/L-band_analysis/L_band_target_pairs.lst'

    with open(todo_list) as files_todo:
        stars_todo = files_todo.readlines()

    #------------------------------------
    #Loop over todo files

    for star in stars_todo:

        #------------------------------------
        #Check list of already done

        done_list = '/mnt_'+node+'/datax/users/eenriquez/L-band_analysis/L_band_processed_targets.lst'

        with open(done_list) as file_done:
            stars_done = file_done.readlines()

        #------------------------------------
        # Add file name to done list, skip if already there.

        if star in stars_done:
            continue
        else:
            with open(done_list,'a') as file_done:
                file_done.write(star)

        #------------------------------------
        #Naming

        star_path = star.split('spliced')[0]
        star_name = 'spliced'+star.replace('.fil','.h5').split('spliced')[-1]

        #------------------------------------
        #Run turbo_seti

        out,err = reset_outs()

        command=['python','/home/eenriquez/software/bl-soft/turbo_seti/bin/seti_event.py',star,'-M','2','-c','115,395']
        print ' '.join(command)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = proc.communicate()

        with open(out_dir+star_name.replace('.h5','.seti_event.log'), 'a') as f:
            f.write(out)

        if err or proc.returncode != 0:
            with open(out_dir+star_name.replace('.h5','.seti_event.err'), 'a') as f:
                f.write(err)

            #------------------------------------
            #Add failed one to list.

            error_list = '/mnt_'+node+'/datax/users/eenriquez/L-band_analysis/L_band_failed_targets.lst'

            with open(done_list,'a') as file_fail:
                file_fail.write(star)

if __name__ == "__main__":
    cmd_tool()
