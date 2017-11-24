# header etc
import pandas as pn
import numpy as np
import ggplot

#def trace(record, keys, timekey, limits):
#    ...

def per_stage_trace(record, keys, limits=""):
    """Create a plot of selected per-stage Record elements."""
    start_stage = limits[0] if limits else 0
    end_stage = limits[1] if limits else record["n_stages"]
    stages = np.arange(start_stage, end_stage)
    # Make data frame
    keydict = dict([(key, record[key][stages]) for key in keys])
    keydict["stage"] = stages
    df = pd.DataFrame(keydict)
    # Make plotting object
    g <- ggplot.ggplot(aes(x="stage"), data=keydict)
    for k in keys: g += geom_line(aes(y=key))
    # TODO: add colour, linetype, line vs point etc options?
    return g


def plot_population_resources(record, limits=""):
    return per_stage_trace(record, ["population_size", "resources"], limits)

