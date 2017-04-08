#!/usr/bin/env python

from argparse import ArgumentParser
import socket
import subprocess
from blimpy import Waterfall
from optparse import OptionParser
import sys
import os

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
    p.add_option('-o', '--out_dir', dest='out_dir', type='str', default='/datax/users/eenriquez/L-band_analysis/', help='Location for output files. Default: local dir. ')
    p.add_option('-n', '--node', dest='node', type='str', default=local_host, help='Name of host node.')
    opts, args = p.parse_args(sys.argv[1:])

    out_dir = opts.out_dir
    node = opts.node

    #------------------------------------
    #Check if running locally

    if node == local_host:
        extra_path = ''
    else:
        extra_path = '/mnt_'+node

    #------------------------------------
    #Check list of files todo

    todo_list = extra_path+out_dir+'L_band_target_pairs.lst'

    with open(todo_list) as files_todo:
        stars_todo = files_todo.readlines()

    #------------------------------------
    #Loop over todo files

    for star in stars_todo:
        #------------------------------------
        #Check list of already done

        done_list = extra_path+out_dir+'L_band_processed_targets.lst'

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
        #Naming, and choosing the right path.

        star_name = 'spliced'+star.split('spliced')[-1].rstrip()

        if node == local_host:
            star_path = '/datax'+star.split('/datax')[-1].split('spliced')[0].rstrip('/')+'/'
        else:
            star_path = star.split('spliced')[0].rstrip('/')+'/'

        #------------------------------------
        #Run turbo_seti

        out,err = reset_outs()

        command=['python','/home/eenriquez/software/bl-soft/turbo_seti/bin/seti_event.py',star_path+star_name,'-M','2','-s','20','-c','115,395','-o',out_dir+'stars_out/']
        print ' '.join(command)

        err_file = out_dir+'logs/'+star_name.replace('.fil','.seti_event.err')
        out_file = out_dir+'logs/'+star_name.replace('.fil','.seti_event.log')

        try_it = os.system(' '.join(command) +' 2> '+err_file+' 1> '+out_file)

#         proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         (out, err) = proc.communicate()
#
#         with open(out_dir+'logs/'+star_name.replace('.fil','.seti_event.log'), 'a') as f:
#             f.write(out)
#
#         if err or proc.returncode != 0:
#             with open(out_dir+'logs/'+star_name.replace('.fil','.seti_event.err'), 'a') as f:
#                 f.write(err)

        #------------------------------------
        #Add failed one to list.

        if os.path.isfile(err_file):
            error_list = extra_path+out_dir+'L_band_failed_targets.lst'

            with open(error_list,'a') as file_fail:
                file_fail.write(star)

        #------------------------------------
        #Delete h5 file

        os.remove(out_dir+'stars_out/'+star_name.replace('.fil','.h5'))


if __name__ == "__main__":
    cmd_tool()
