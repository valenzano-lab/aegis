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
# - add missing plots

# TODO averaging: function separate from make_dataframe

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
            self.plot_methods = ["plot_population_resources",\
                                 "plot_starvation",\
                                 "plot_fitness",\
                                 "plot_fitness_term",\
                                 "plot_fitness_term_overlay",\
                                 "plot_entropy_gt",\
                                 "plot_entropy_bits",\
                                 "plot_age_distribution",\
                                 "plot_density_per_locus",\
                                 "plot_density",\
                                 "plot_mean_gt",\
                                 "plot_var_gt",\
                                 "plot_entropy_gt",\
                                 "plot_repr_value",\
                                 "plot_mean_repr",\
                                 "plot_cmv_surv",\
                                 "plot_n1",\
                                 "plot_n1_var",\
                                 "plot_actual_death_rate",\
                                 "plot_density_snap_overlay",\
                                 "plot_repr_value_snap_overlay",\
                                 "plot_cmv_surv_snap_overlay"
                                 ]
            self.plot_names = ["pop-res",\
                               "starvation",\
                               "fitness",\
                               "fitness_term",\
                               "per_fitness_term_overlay",\
                               "plot_entropy_gt",\
                               "plot_entropy_bits",\
                               "age_distribution",\
                               "density_per_locus",\
                               "density",\
                               "mean_gt",\
                               "var_gt",\
                               "entropy_gt",\
                               "repr_value",\
                               "mean_repr",\
                               "cmv_surv",\
                               "n1",\
                               "n1_var",\
                               "actual_death_rate",\
                               "density_overlay",\
                               "repr_value_overlay",\
                               "cmv_surv_overlay"
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

    # TODO check if pandas.melt does a better job with this
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

    def dataframe_mean(self, df, key1, key2, key3, window):
        """
        Firstly select only rows of dataframe for which column key3 is in window.
        Group dataframe by columns key1.
        Get the mean of the groups for the key1 column.
        Return a pandas.DataFrame."
        """
        for key in [key1, key2, key3]:
            if not isinstance(key, str):
                raise ValueError("Invalid key value: {}".format(snapshot))
        if not isinstance(window,list):
                raise ValueError("Invalid windows value: {}".format(snapshot))
        return df[df[key3].isin(window)].groupby(key1)[key2].mean().reset_index()

    ##################
    # plot functions #
    ##################
    def solo_plot(self, keys, dimlabels, xaxis, geoms, subkey="", snapshot="",\
            overlay=False, title = ""):
        """Create a single plot, either of multiple data series (overlay=False)
        or of overlayed slices of a single series (overlay=True)."""
        data = self.make_dataframe(keys, dimlabels, subkey, snapshot)
        #print data
        #dimlabels = data.columns.tolist()
        #if "key" in dimlabels: dimlabels.remove("key")
        # check snapshot valid and set it
        if not isinstance(snapshot,str) and not isinstance(snapshot,int):
            raise ValueError("Invalid snapshot value: {}".format(snapshot))
        elif snapshot == "all":
            snapshot = -1
        elif snapshot == "":
            snapshot = self.record["number_of_snapshots"]-1
        elif snapshot < -1 or snapshot >= self.record["number_of_snapshots"]:
            raise ValueError("Invalid snapshot value: {}".format(snapshot))

        # add snapshot info to title
        if snapshot>-1:
            title += "\n(snapshot: {})".format(snapshot) # +1?

        if overlay: data[dimlabels[0]] = data[dimlabels[0]].astype(str)
        plot = ggplot.ggplot(data, ggplot.aes(x=xaxis,
            y="value", color = "snapshot" if overlay else "key"))
        for g in geoms:
            plot += getattr(ggplot, "geom_"+g)()
        plot += ggplot.labs(title=title)
        return plot

    # TODO make this work and make use of it
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
    def stage_trace(self, keys, title=""):
        """Simple per-stage line trace for specified Record keys."""
        return self.solo_plot(keys, ["stage"], "stage", ["step"], "", "", False,\
                title)

    def snapshot_trace(self, keys, subkey="", title=""):
        """Simple per-snapshot line + point trace for specified Record keys."""
        return self.solo_plot(keys, ["snapshot"], "snapshot", ["line","point"],\
                subkey, "all", False, title)

    def age_trace(self, keys, geoms, snapshot="", title=""):
        """Simple per-age trace for specified Record keys for specified snapshot.
        (defualt: last snapshot)"""
        if snapshot == "all":
            raise ValueError("Invalid snapshot value: 'all'. Use \
                    age_overlay instead.")
        return self.solo_plot(keys, ["snapshot","age"], "age", geoms, "", snapshot,\
                False, title)

    # TODO add age_trace_overlay (regarding snapshots)

    def snapshot_overlay(self, keys, dimlabel, subkey="", title=""):
        """Per-snapshot overlay line plot of a single Record key."""
        return self.solo_plot(keys, ["snapshot", dimlabel], dimlabel, ["line"],\
                subkey, "all", True, title)

    ###############
    # stage_trace #
    ###############
    def plot_population_resources(self):
        # TODO: enable swapping order of series for constant v variable resources
        # TODO: Set limits at object level?
        return self.stage_trace(["resources","population_size"], "Population and resources")

    def plot_starvation(self):
        # TODO: plot y-axis in log-space of base matching starvation increment
        return self.stage_trace(["surv_penf","repr_penf"], "Survival and reproduction penalty factors upon starvation")

    ##################
    # snapshot_trace #
    ##################
    def plot_fitness(self):
        return self.snapshot_trace(["fitness", "junk_fitness"], "", "Fitness")

    def plot_entropy_gt(self):
        return self.snapshot_trace(["entropy_gt"], "", "")

    def plot_entropy_bits(self):
        return self.snapshot_trace(["entropy_bits"], "", "")

    #############
    # age_trace #
    #############
    def plot_repr_value(self):
        return self.age_trace(["repr_value","junk_repr_value"], ["point","line"], \
                "", "Reproduction value")

    def plot_mean_repr(self):
        return self.age_trace(["mean_repr","junk_repr"], ["point","line"],\
                "", "Genome-encoded reproduction rate")

    def plot_cmv_surv(self):
        return self.age_trace(["cmv_surv","junk_cmv_surv"], ["point","line"],\
                "", "Cumulative genome-encoded survival rate")

    def plot_fitness_term(self):
        return self.age_trace(["fitness_term","junk_fitness_term"], ["point",\
                "line"], "", "Fitness term")

    ####################
    # snapshot_overlay #
    ####################
    # TODO should we also plot age_distribution averaged over a window around
    # snapshot stage?
    def plot_age_distribution(self):
        return self.snapshot_overlay(["snapshot_age_distribution"], "age", "",\
                "Age distribution")

    def plot_fitness_term_overlay(self):
        return self.snapshot_overlay(["fitness_term"], "age", "",\
                "Fitness term overlay")

    def plot_density_snap_overlay(self):
        return self.snapshot_overlay(["density"], "genotype", ["a"], "Density")

    def plot_repr_value_snap_overlay(self):
        return self.snapshot_overlay(["repr_value"], "age", "", "Reproduction value")

    def plot_cmv_surv_snap_overlay(self):
        return self.snapshot_overlay(["cmv_surv"], "age", "", "Cumulative survival")

    ############
    # specific #
    ############

    # arrays

    def plot_n1(self, snapshot=""):
        plot = self.solo_plot(["n1"], ["snapshot","bit"], "bit", ["point"],\
                "", snapshot, False, "Distribution of 1's per bit")
        surv_repr = self.record["max_ls"]*self.record["n_base"]
        repr_neut = (2*self.record["max_ls"]-self.record["maturity"])\
                *self.record["n_base"]
        plot += ggplot.geom_vline(x=[surv_repr, repr_neut], color = "black",\
                linetype="dashed")
        return plot

    def plot_n1_var(self, snapshot=""):
        plot = self.solo_plot(["n1_var"], ["snapshot","bit"], "bit", ["point"],\
                "", snapshot, False, "Variation of distribution of 1's per bit")
        surv_repr = self.record["max_ls"]*self.record["n_base"]
        repr_neut = (2*self.record["max_ls"]-self.record["maturity"])\
                *self.record["n_base"]
        plot += ggplot.geom_vline(x=[surv_repr, repr_neut], color = "black",\
                linetype="dashed")
        return plot

    # dictionaries
    # auxuliary funtion
    def check_asrn(self, key):
        if key not in ['a','s','r','n']:
            raise ValueError("Invalid key value: {}".format(key))

    # TODO what should be the x-axis?
    # now it prints all 21 y values for locus x on (x,y)
    # plot it agains index (like what I draw on paper)
    def plot_density_per_locus(self, subkey="a", snapshot=""):
        """
        subkey: 'a','s','r','n'.
        snapshot: last by default
        """
        self.check_asrn(subkey)
        return self.solo_plot(["density_per_locus"], ["snapshot", "locus", \
                "genotype"], "locus", ["point"], list(subkey), snapshot, False,\
                "")

    def plot_density(self, subkey="a", snapshot=""):
        """
        subkey: 'a','s','r','n'.
        snapshot: last by default
        """
        self.check_asrn(subkey)
        return self.solo_plot(["density"], ["snapshot", "genotype"], "genotype",\
                ["point","line"], list(subkey), snapshot, False,\
                "Density")

    def plot_mean_gt(self, subkey="a", snapshot=""):
        self.check_asrn(subkey)
        plot = self.solo_plot(["mean_gt"], ["snapshot", "locus"], "locus",\
                ["point"], list(subkey), snapshot, False,\
                "Mean genotype value")
        surv_repr = self.record["max_ls"]
        repr_neut = (2*self.record["max_ls"]-self.record["maturity"])
        plot += ggplot.geom_vline(x=[surv_repr, repr_neut], color = "black",\
                linetype="dashed")
        return plot

    def plot_var_gt(self, subkey="a", snapshot=""):
        self.check_asrn(subkey)
        plot = self.solo_plot(["var_gt"], ["snapshot", "locus"], "locus",\
                ["point"], list(subkey), snapshot, False,\
                "Genotype value variance")
        surv_repr = self.record["max_ls"]
        repr_neut = (2*self.record["max_ls"]-self.record["maturity"])
        plot += ggplot.geom_vline(x=[surv_repr, repr_neut], color = "black",\
                linetype="dashed")
        return plot

    def plot_entropy_gt(self, subkey="a"):
        self.check_asrn(subkey)
        return self.solo_plot(["entropy_gt"], ["snapshot"], "snapshot",\
                ["point"], list(subkey), "all", False,\
                "")

    def plot_actual_death_rate(self, window_size=100):
        # check window size is OK
        if window_size*(self.record["number_of_snapshots"]+1) > \
                self.record["number_of_stages"]:
            raise ValueError("Window size is too big; overlap.")
        windows = [range(window_size)]
        window_size /= 2
        for s in self.record["snapshot_stages"][1:-1]:
            windows += [range(s-window_size,s+window_size)]
        windows += [range(self.record["number_of_stages"]-\
                window_size, self.record["number_of_stages"])]

        data = self.make_dataframe(["actual_death_rate"], ["stage","age"])

        mean_data = pd.DataFrame()
        for i in range(len(windows)):
            x = self.dataframe_mean(data,"age","value","stage",windows[i])
            #x["snapshot_stage"] = self.record["snapshot_stages"][i]
            x["snapshot_stage"] = i
            mean_data = mean_data.append(x)

        mean_data["snapshot_stage"] = mean_data["snapshot_stage"].astype(str)
        plot = ggplot.ggplot(mean_data, ggplot.aes(x="age",
            y="value", color = "snapshot_stage"))
        #plot += ggplot.geom_point()
        plot += ggplot.geom_line()
        plot += ggplot.labs(title="Actual death rate")
        return plot
