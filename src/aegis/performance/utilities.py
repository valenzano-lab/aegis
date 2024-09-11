def profile_manually(function):
    import line_profiler

    profiler = line_profiler.LineProfiler()
    profiler.add_function(function)
    profiler.runcall(function)
    profiler.print_stats()
