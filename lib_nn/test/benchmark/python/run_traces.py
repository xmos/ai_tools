
import os
import argparse
import subprocess
from process_trace import process_trace
from lib_nn_funcs import *

TRACE_FILE = 'trace.local.tmp'
XSIM_CMD = ['xsim', '--trace-to', None, '--enable-fnop-tracing', '--args']



def run_single(func_name, args, xe_argset, fnames = None):

    fnames = [func_name] if fnames is None else fnames

    tf_path = os.path.join(args.out_dir, TRACE_FILE)
    op_args = [func_name, *[str(x) for x in xe_argset]]
    xcmd = [*XSIM_CMD, *op_args]
    xcmd[2] = tf_path
    # print("\t{}".format(" ".join(op_args)))
    # print(xcmd)
    subprocess.call(xcmd)
    return process_trace(fnames, tf_path)


def run_traces(args, func_name):
    print("Running traces for {}..".format(func_name))

    if func_name not in FUNC_LUT:
        raise Exception("Function {} not found in LUT.".format(func_name))

    proc_func = FUNC_LUT[func_name]

    meas = lambda rg, fnames=None: run_single(func_name, args, rg, fnames)
    proc_func(meas, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('xe_file', help='.xe file to be traced.')
    parser.add_argument('funcs_to_test', nargs='*', help='Functions to be tested.')

    parser.add_argument('-d', '--out-dir', dest='out_dir', default='.', help='Directory to put results.')
    parser.add_argument('-s', '--show-plot', action='store_true', help='Display plot, rather than saving it.')

    args = parser.parse_args()

    XSIM_CMD.append(args.xe_file)

    for func in args.funcs_to_test:
        run_traces(args, func)