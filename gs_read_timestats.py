#!/usr/bin/env python

import cProfile, pstats, StringIO, argparse, os

parser = argparse.ArgumentParser(description='Display cProfile time \
        statistics from a profiled simulation run.')
parser.add_argument('dir', help="path to simulation directory")
parser.add_argument('-f', metavar="<str>", default="timestats.txt",
        help="path to statistics file within directory (default: timestats.txt)")
parser.add_argument('-s', metavar='<str>', default="tottime",
        help="sort stats by specified column (default: tottime)")
parser.add_argument('-c', '--callees', metavar='<str>', default="",
        help="show statistics for all functions called by specified function")
parser.add_argument('-C', '--callers', metavar='<str>', default="",
        help="show statistics for all functions calling specified function")
parser.add_argument('-n', type=int, metavar="<int/float>", default=20,
        help="return only <n> (if integer) or <n>*100%% (if float)\
        top entries from stats list (default: 20)")
parser.add_argument('-A', '--all', action="store_false",
        help="include results from all processes \
                (default: only processes from gs_ files)")

args = parser.parse_args()
s = StringIO.StringIO()

keyword = "gs_" if args.all else None
os.chdir(args.dir)
ps = pstats.Stats(args.f, stream=s)
ps.strip_dirs() # remove the path from module names
ps.sort_stats(args.s)
if args.callees != "":
    ps.print_callees(args.callees, args.n)
elif args.callers != "":
    ps.print_callers(args.callers, args.n)
else:
    ps.print_stats(keyword, args.n) # print top_n stats

print s.getvalue()
