#!/usr/bin/env python

# Import libraries and define arguments
import argparse,os,sys,warnings,pyximport; pyximport.install()
from gs_core import Simulation
try:
    import Cpickle as picke
except:
    import pickle

parser = argparse.ArgumentParser(description='Run the genome ageing \
        simulation.')
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-o', metavar="<str>", default="output",
        help="prefix of simulation output file (default: output)")
parser.add_argument('-l', metavar="<str>", default="log",
        help="prefix of simulation log file (default: log)")
parser.add_argument('-s', default="",
        help="path to simulation seed file (default: no seed)")
parser.add_argument('-S', default=-1, 
        help="Run number in seed file from which to take seed population;\
                -1 indicates to seed each new run with the corresponding\
                run from the seed file (default: -1)")
parser.add_argument('-c', metavar='<str>', default="config",
        help="name of configuration file within simulation directory \
                (default: config.py)")
parser.add_argument('-r', type=int, metavar="<int>", default=10,
        help="report information every <int> stages (default: 10)")
parser.add_argument('-p', '--profile', action="store_true",
        help="profile genome simulation with cProfile")
parser.add_argument('-v', '--verbose', action="store_true",
        help="display full information at each report stage \
                (default: only starting population)")
args = parser.parse_args()

if args.profile:
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable() # start profiling

if args.s != "":
    args.s = os.path.abspath(args.s) # Get abspath before changing dir
#Change to simulation directory
try:
    sys.path.remove(os.getcwd())
    os.chdir(args.dir)
    sys.path = [os.getcwd()] + sys.path
except OSError:
    exit("Error: Specified simulation directory does not exist.")
    pr.create_stats()
with warnings.catch_warnings(DeprecationWarning):
    sim = Simulation(args.c, args.s, args.S, args.r, args.verbose)
    sim.execute()
    sim.finalise(args.o, args.l)

if args.profile:
    pr.dump_stats('timestats.txt')
