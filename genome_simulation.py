#!/usr/bin/env python

# Import libraries and define arguments
import argparse,os,sys,pyximport; pyximport.install()
from gs_core import Simulation

parser = argparse.ArgumentParser(description='Run the genome ageing \
        simulation.')
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-o', metavar="<str>", default="output",
        help="prefix of simulation output file (default: output)")
parser.add_argument('-l', metavar="<str>", default="log",
        help="prefix of simulation log file (default: log)")
parser.add_argument('-s', metavar="<str>", default="",
        help="path to population seed file (default: no seed)")
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

seed = args.s
#Change to simulation directory
try:
    sys.path.remove(os.getcwd())
    os.chdir(args.dir)
    sys.path = [os.getcwd()] + sys.path
except OSError:
    exit("Error: Specified simulation directory does not exist.")

sim = Simulation(args.c, seed, args.r, args.verbose)
sim.execute()
sim.finalise(args.o, args.l)

if args.profile:
    pr.create_stats()
    pr.dump_stats('timestats.txt')
