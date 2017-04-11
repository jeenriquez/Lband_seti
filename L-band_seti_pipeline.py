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
    """ Command line tool for L-band-seti-pipeline. """

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
    #moving to output loc.
    os.chdir(out_dir+'stars_out/')

    #------------------------------------
    #Loop over todo files

    for star in stars_todo:
        #------------------------------------
        #Naming, and choosing the right path.

        star_name = 'spliced'+star.split('spliced')[-1].rstrip()

        if node == local_host:
            star_path = '/datax'+star.split('/datax')[-1].split('spliced')[0].rstrip('/')+'/'
        else:
            star_path = star.split('spliced')[0].rstrip('/')+'/'

        #------------------------------------
        #Check list of bad data
        bad_list = '/home/eenriquez/software/Lband_seti/bad_data.lst'

        with open(bad_list) as file_bad_list:
            stars_file_bad_list = file_bad_list.readlines()

        #------------------------------------
        # Skip if bad data.

        if star_name in stars_file_bad_list:
            print star_name +' is a baaaad star name.'
            continue

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
        # Make hdf5

        if not os.path.isfile(star_name.replace('.fil','.h5')):

            out,err = reset_outs()

            command=['python','/home/eenriquez/software/Lband_seti/fil2h5.py',star_path+star_name]
            print ' '.join(command)
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = proc.communicate()

            err_file = out_dir+'logs/'+star_name.replace('.fil','.make_h5.err')
            out_file = out_dir+'logs/'+star_name.replace('.fil','.make_h5.log')

            with open(out_file, 'a') as f:
                f.write(out)

            if err or proc.returncode != 0:
                with open(err_file, 'a') as f:
                    f.write(err)

        #------------------------------------
        #Run turbo_seti

        out,err = reset_outs()

        command=['python','/home/eenriquez/software/bl-soft/turbo_seti/bin/seti_event.py',star_name.replace('.fil','.h5'),'-M','2','-s','20','-c','115,395']
        print ' '.join(command)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = proc.communicate()

        err_file = out_dir+'logs/'+star_name.replace('.fil','.seti_event.err')
        out_file = out_dir+'logs/'+star_name.replace('.fil','.seti_event.log')
#         try_it = os.system(' '.join(command) +' 2> '+err_file+' 1> '+out_file)

        with open(out_file, 'a') as f:
            f.write(out)

        if err or proc.returncode != 0:
            with open(err_file, 'a') as f:
                f.write(err)

        #------------------------------------
        #Add failed one to list.

        remove_h5=True

        if os.path.isfile(err_file):
            if os.path.getsize(err_file) > 0:
                error_list = extra_path+out_dir+'L_band_failed_targets.lst'

                with open(error_list,'a') as file_fail:
                    file_fail.write(star)

                remove_h5 =False
            else:
                os.remove(err_file)

        #------------------------------------
        #Delete h5 file

        if remove_h5:
            os.remove(out_dir+'stars_out/'+star_name.replace('.fil','.h5'))

if __name__ == "__main__":
    cmd_tool()
