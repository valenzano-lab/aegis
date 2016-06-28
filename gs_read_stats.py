#!/usr/bin/env python
# reads the gs_stats file that was output by genome_simulation.py in the
# simulation directory

import cProfile, pstats, StringIO

s = StringIO.StringIO()

prf = 'gs_stats' # profile filename
sortby = 'tottime'
top_n = 20 # integer or percentile
keyword = 'gs_' # to pattern match the standard name that is printed, otherwise =None

ps = pstats.Stats(prf, stream=s)
ps.strip_dirs() # remove the path from module names
ps.sort_stats(sortby)
ps.print_stats(keyword, top_n) # print top_n stats

print s.getvalue()
