########################################################################
# AEGIS - Ageing of Evolving Genomes in Silico                         #
# Module: Plot                                                         #
# Classes: APlot [better name tbc]                                     #
# Description: Wrapper object that takes and stores a passed Record    #
#   object and implements plotting methods on it.                      #
########################################################################
# TODO:
# - Formatting code (e.g. theme_bw)
# - Overlaying/averaging multiple runs
# - Restore code for limits
# - Write tests
# - add titles, axis labels, snapshot info...
# - add missing plots

import numpy as np, pandas as pd, ggplot, os, shutil
try:
       import cPickle as pickle
except:
       import pickle

class Plotter:
    """Wrapper class for storing a Record and its associated plots."""

    def __init__(self, record):
        """Import a Record object and initialise plotting methods."""
        rfile = open(record, "rb")
        try:
            self.record = pickle.load(rfile)
            self.plot_methods = [#"plot_population_resources",\
                                 #"plot_starvation",\
                                 #"plot_fitness",\
                                 #"plot_entropy_gt",\
                                 #"plot_entropy_bits",\
                                 #"plot_age_distribution",\
                                 #"plot_density_per_locus",\
                                 #"plot_per_age_fitness",\
                                 #"plot_density",\
                                 "plot_mean_gt",\
                                 "plot_var_gt",\
                                 "plot_entropy_gt",\
                                 "plot_repr_value"
                                 ]
            self.plot_names = [#"pop-res",\
                               #"starvation",\
                               #"fitness",\
                               #"plot_entropy_gt",\
                               #"plot_entropy_bits",\
                               #"age_distribution",\
                               #"density_per_locus",\
                               #"per_age_fitness",\
                               #"density",\
                               "mean_gt",\
                               "var_gt",\
                               "entropy_gt",\
                               "repr_value"
                               ]
            self.plots = []
        finally:
            rfile.close()

    def generate_plots(self):
        for m in self.plot_methods:
            self.plots.append(getattr(self, m)())

    def save_plots(self):
        # Make/replace output directory
        pm,pn,p = self.plot_methods, self.plot_names, self.plots
        if not len(pm) == len(pn) == len(p):
            errstr = "Plot names, methods and images are of different lengths."
            raise ValueError(errstr)
        outdir = self.record["output_prefix"] + "_plots"
        if os.path.exists(outdir): # Overwrite existing output
                shutil.rmtree(outdir)
        os.makedirs(outdir)
        for n in xrange(len(self.plots)):
            outpath = os.path.join(outdir, self.plot_names[n] + ".png")
            self.plots[n].save(outpath)

    def get_flattened_coords(self, dim, arr):
        """Given an array and a dimension, returns the original
        co-ordinate in that dimension of each element in a flattened
        version of the input array."""
        values = np.arange(arr.shape[dim]) # Set of possible co-ord values
        shape = [int(np.prod(arr.shape[dim+1:])),
                int(np.prod(arr.shape[:dim+1]))/arr.shape[dim]]
        return np.tile(values, shape).T.flatten().astype(int)

    def make_dataframe(self, keys, dimlabels, subkeys="", snapshot="", source=""):
        """
        Convert one or more Record entries into a data-frame, with
        columns for the corresponding key, the value, and the
        co-ordinates in the entry array.

                     subkeys <list> - if entry is a dictionary, this determines
                                      which keys will be included in the data frame
               snapshot <int/'all'> - if entry has snapshot dimension, this
                                      determines which respecting indices will be
                                      included in the data frame (defualt: last)
        source <dict/numpy.ndarray> - self.record by default
        """

        # check subkeys valid
        if subkeys:
            if not isinstance(subkeys, list):
                raise TypeError("subkeys is of invalid type: {}. Must be list.".\
                        format(type(subkeys)))
            elif not all(isinstance(k, str) for k in subkeys):
                raise TypeError("subkeys elements are of invalid type: {}. Must be\
                        string".\
                        format(type(subkeys[0])))
        # check snapshot valid and set it
        if not isinstance(snapshot,str) and not isinstance(snapshot,int):
            raise ValueError("Invalid snapshot value: {}".format(snapshot))
        elif snapshot == "all":
            snapshot = -1
        elif snapshot == "":
            snapshot = self.record["number_of_snapshots"]-1
        elif snapshot < -1 or snapshot >= self.record["number_of_snapshots"]:
            raise ValueError("Invalid snapshot value: {}".format(snapshot))
        # set source
        source = source if isinstance(source,dict) or isinstance(source,np.ndarray)\
                else self.record

        # build data frame
        df = pd.DataFrame()
        for key in keys:

            val = source[key]
            # if dictionary
            if isinstance(val, dict):
                subkeys = val.keys() if not subkeys else subkeys
                df = df.append(self.make_dataframe(subkeys, dimlabels,
                    "", snapshot, val))

            # if numpy.ndarray
            else:
                ddict = {"key":key, "value":val.flatten()}
                for n in xrange(len(dimlabels)):
                    ddict[dimlabels[n]] = self.get_flattened_coords(n, val)
                df = df.append(pd.DataFrame(ddict))
                if self.record["number_of_snapshots"] in val.shape and snapshot>-1:
                    df = df[df.snapshot == snapshot]

        return df

    ##################
    # plot functions #
    ##################
    def solo_plot(self, keys, dimlabels, xaxis, geoms, subkey="", snapshot="",\
            overlay=False):
        """Create a single plot, either of multiple data series (overlay=False)
        or of overlayed slices of a single series (overlay=True)."""
        data = self.make_dataframe(keys, dimlabels, subkey, snapshot)
        #print data
        #dimlabels = data.columns.tolist()
        #if "key" in dimlabels: dimlabels.remove("key")
        if overlay: data[dimlabels[0]] = data[dimlabels[0]].astype(str)
        plot = ggplot.ggplot(data, ggplot.aes(x=xaxis,
            y="value", color = "snapshot" if overlay else "key"))
        for g in geoms:
            plot += getattr(ggplot, "geom_"+g)()
        return plot

    def grid_plot(self, keys, dimlabels, facet, geoms, subkey=""):
        """Create a grid of plots, showing corresponding slices of one
        or more data series (e.g. at different time points."""
        data = self.make_dataframe(keys, dimlabels, subkey)
        xlabel = dimlabels[1] if dimlabels[0] == facet else dimlabels[0]
        plot = ggplot.ggplot(data, ggplot.aes(x=xlabel, y="value",
            color = "key")) + ggplot.facet_wrap(facet)
        for g in geoms:
            plot += getattr(ggplot, "geom_"+g)()
        return plot

    ##############
    # plot types #
    ##############

    def stage_trace(self, keys):
        """Simple per-stage line trace for specified Record keys."""
        return self.solo_plot(keys, ["stage"], "stage", ["step"])

    def snapshot_trace(self, keys, subkey=""):
        """Simple per-snapshot line + point trace for specified Record keys."""
        return self.solo_plot(keys, ["snapshot"], "snapshot", ["line","point"],\
                subkey, "all")

    def age_trace(self, keys, geoms, snapshot=""):
        """Simple per-age trace for specified Record keys for specified snapshot.
        (defualt: last snapshot)"""
        if snapshot == "all":
            raise ValueError("Invalid snapshot value: 'all'. Use \
                    age_overlay instead.")
        return self.solo_plot(keys, ["snapshot","age"], "age", geoms, "", snapshot)

    # TODO add age_overlay (wrg to snapshot)

    def snapshot_overlay(self, keys, dimlabel, subkey=""):
        """Per-snapshot overlay line plot of a single Record key."""
        return self.solo_plot(keys, ["snapshot", dimlabel], dimlabel, ["line"],\
                subkey, "all", True)

    ###############
    # stage_trace #
    ###############
    def plot_population_resources(self):
        # TODO: enable swapping order of series for constant v variable resources
        # TODO: Set limits at object level?
        return self.stage_trace(["population_size", "resources"])

    def plot_starvation(self):
        # TODO: plot y-axis in log-space of base matching starvation increment
        return self.stage_trace(["surv_penf","repr_penf"])

    ##################
    # snapshot_trace #
    ##################
    def plot_fitness(self):
        return self.snapshot_trace(["fitness", "junk_fitness"])

    def plot_entropy_gt(self):
        return self.snapshot_trace(["entropy_gt"])

    def plot_entropy_bits(self):
        return self.snapshot_trace(["entropy_bits"])

    #############
    # age_trace #
    #############
    def plot_repr_value(self):
        return self.age_trace(["repr_value"], ["point","line"])

    def plot_junk_repr_value(self):
        return self.age_trace(["junk_repr_value"], ["point","line"])

    def plot_mean_repr(self):
        return self.age_trace(["mean_repr"], ["point","line"])

    def plot_junk_repr(self):
        return self.age_trace(["junk_repr"], ["point","line"])

    def plot_cmv_surv(self):
        return self.age_trace(["cmv_surv"], ["point","line"])

    def plot_fitness_term(self):
        return self.age_trace(["fitness_term"], ["point","line"])

    def plot_junk_fitness_term(self):
        return self.age_trace(["junk_fitness_term"], ["point","line"])

    ####################
    # snapshot_overlay #
    ####################
    def plot_age_distribution(self):
        return self.snapshot_overlay(["snapshot_age_distribution"], "age")

    def plot_per_age_fitness(self):
        return self.snapshot_overlay(["fitness_term"], "age")

    ############
    # specific #
    ############

    # dictionaries
    # TODO why do I get that this function is not defined if I use it below?
    def check_asrn(key):
        if subkey not in ['a','s','r','n']:
            raise ValueError("Invalid key value: {}".format(subkey))

    def plot_density_per_locus(self, subkey="a", snapshot=""):
        """
        subkey: 'a','s','r','n'.
        snapshot: last by default
        """
        #check_asrn(subkey)
        return self.solo_plot(["density_per_locus"], ["snapshot", "locus", \
                "genotype"], "locus", ["point"], list(subkey), snapshot)

    def plot_density(self, subkey="a", snapshot=""):
        """
        subkey: 'a','s','r','n'.
        snapshot: last by default
        """
        #check_asrn(subkey)
        return self.solo_plot(["density"], ["snapshot", "genotype"], "genotype",\
                ["point","line"], list(subkey),snapshot)

    def plot_mean_gt(self, subkey="a", snapshot=""):
        #check_asrn(subkey)
        return self.solo_plot(["mean_gt"], ["snapshot", "locus"], "locus",\
                ["point"], list(subkey),snapshot)

    def plot_var_gt(self, subkey="a", snapshot=""):
        #check_asrn(subkey)
        return self.solo_plot(["var_gt"], ["snapshot", "locus"], "locus",\
                ["point"], list(subkey),snapshot)

    def plot_entropy_gt(self, subkey="a"):
        #check_asrn(subkey)
        return self.solo_plot(["entropy_gt"], ["snapshot"], "snapshot",\
                ["point"], list(subkey), "all")
