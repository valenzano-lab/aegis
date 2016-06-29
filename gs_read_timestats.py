#!/usr/bin/env python

import cProfile, pstats, StringIO, argparse, os

parser = argparse.ArgumentParser(description='Display cProfile time \
        statistics from a profiled simulation run.')
parser.add_argument('file', help="path to stats file")
parser.add_argument('-s', metavar='<str>', default="tottime",
        help="sort stats by specified column (default: tottime)")
parser.add_argument('-c', '--callees', metavar='<str>', default="",
        help="show statistics for all functions called by specified function")
parser.add_argument('-C', '--callers', metavar='<str>', default="",
        help="show statistics for all functions calling specified function")
parser.add_argument('-n', type=int, metavar="<int/float>", default=20,
        help="return only <n> (if integer) or <n>*100%% (if float)\
        top entries from stats list (default: 20)")
parser.add_argument('-T', action="store_true",
        help="hide overall profiling stats (default: False")
parser.add_argument('-A', '--all', action="store_true",
        help="include results from all processes \
                (default: only processes from gs_ files)")

args = parser.parse_args()
s = StringIO.StringIO()

keyword = "gs_" if not args.all else None
ps = pstats.Stats(args.file, stream=s)
ps.strip_dirs() # remove the path from module names
ps.sort_stats(args.s)
if not args.T:
    ps.print_stats(keyword, args.n) # print top_n stats
if args.callees != "":
    ps.print_callees(args.callees, args.n)
if args.callers != "":
    ps.print_callers(args.callers, args.n)

print s.getvalue()
