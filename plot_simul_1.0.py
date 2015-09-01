# lightweight version
# imports
import pygal
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import cPickle
from time import sleep

## input/output paths [string]
file_in = '/home/arian/projects/model/output/plot_values_run1.txt'
file_out = '/home/arian/projects/model/output/'

## unpickle
f = open(file_in,'rb')
pop_in = cPickle.load(f)
n_stage = len(pop_in)-1
res_in = cPickle.load(f)
age_distr_in = cPickle.load(f)
#print np.shape(age_distr_in[0])
repr_rate_in = cPickle.load(f)
repr_rate_sd_in = cPickle.load(f)
for i in range(len(repr_rate_sd_in)):
    repr_rate_sd_in[i] = np.array(repr_rate_sd_in[i])
    repr_rate_sd_in[i].shape = (71,1)
repr_rate_junk_in = cPickle.load(f)
surv_rate_in = cPickle.load(f)
surv_rate_sd_in = cPickle.load(f)
for i in range(len(surv_rate_sd_in)):
    surv_rate_sd_in[i] = np.array(surv_rate_sd_in[i])
    surv_rate_sd_in[i].shape = (71,1)
surv_rate_junk_in = cPickle.load(f)
repr_fit_in = cPickle.load(f)
repr_fit_junk_in = cPickle.load(f)
surv_fit_in = cPickle.load(f)
surv_fit_junk_in = cPickle.load(f)
fit_in = np.array(repr_fit_in)*np.array(surv_fit_in)
fit_junk_in = np.array(repr_fit_junk_in)*np.array(surv_fit_junk_in)
dens_surv_in = cPickle.load(f)
dens_repr_in = cPickle.load(f)
hetrz_mea = cPickle.load(f)
hetrz_mea_sd = cPickle.load(f) # when simul version > 0.6
males_females_ages = cPickle.load(f)
f.close()

# variables
ipx = np.arange(0,71)
nipx = np.linspace(0,70,141)
down_limit_surv = 98
up_limit_repr = 40
snapshot_stages = np.linspace(0,n_stage,16).astype(int)

# scatter survival from two different runs
def scatter_diff_surv(path1, path2, s1=0, s2=n_stage):
    # unpickle
    f1 = open(path1, 'rb')
    f2 = open(path2, 'rb')
    for i in range(7): # surv_rate is 7th in the load order
        surv1 = cPickle.load(f1)
        surv2 = cPickle.load(f2)
    f1.close()
    f2.close()

    var3=s2-s1+1

    var1 = np.zeros((71,))
    var2 = np.zeros((71,))
    for i in range(s1,s2+1):
        var1 += np.array(surv1[i])
        var2 += np.array(surv2[i])
    var1 = var1/var3*100
    var2 = var2/var3*100

    plt.scatter(var1, var2)
    plt.plot((16,16),(0,1),'r--')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlabel('1')
    plt.ylabel('2')
    plt.figtext(0.8, 0.91, 'stage '+str(s1)+'-'+str(s2), size='x-small')
    plt.figtext(0.1, 0.03, '1 = '+path1, size='x-small')
    plt.figtext(0.1, 0.01, '2 = '+path2, size='x-small')
    plt.savefig(file_out+'/scatter_diff_surv.png')
    plt.close()

# actual survival series
def compute_actual_surv_rate(s):
    """
    Takes age distribution of two consecutive stages and computes the
    fractions of those survived from age x to age x+1. The cumulative product
    of those values builds the final result.
    Returns a numpy array.
    """
    div = age_distr_in[s]*pop_in[s]
    div[div == 0] = 1
    stage2 = np.array(list((age_distr_in[s+1]*pop_in[s+1]))[1:]+[0])

    res = stage2 / div
    for i in range(1,len(res)):
        res[i] = res[i-1] * res[i]

    return res

def avr_actual_surv_rate(s):
    """Averages actual survival rate over 100 stages."""
    if s <= 50:
        res = compute_actual_surv_rate(s+100)
        for i in range(s,s+100):
            res += compute_actual_surv_rate(i)
        return res/100
    if s >= n_stage-50:
        res = compute_actual_surv_rate(n_stage-101)
        for i in range(n_stage-100,n_stage-1):
            res += compute_actual_surv_rate(i)
        return res/100
    else:
        res = compute_actual_surv_rate(s+50)
        for i in range(s-50,s+50):
            res += compute_actual_surv_rate(i)
        return res/100

def plot_actual_surv_rate(s):
    """ Plots actual survival rate averaged over 100 stages."""
    y = avr_actual_surv_rate(s)
    plt.scatter(range(71),y)
    plt.plot((16,16),(0,1),'r--')
    plt.title('Actual survival rate')
    plt.xlabel('age')
    #plt.ylabel('N',rotation='horizontal')
    plt.text(60,1.025,'stage '+str(s))
    plt.axis([0,70,0,1])
    plt.savefig(file_out+'/actual_surv_rate.png')
    plt.close()

def snapshot_actual_surv_rate(s1=0,s2=n_stage):
    "Plot 4x4 snapshot plot of actual survival rate, averaged over 100 stages."
    for i in range(1,17):
        y = avr_actual_surv_rate(np.linspace(s1,s2,16).astype(int)[i-1])
        plt.subplot(4,4,i).scatter(range(71),y)
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,1])
        plt.tick_params(labelsize='small')
        plt.text(50,100.01,str(np.linspace(s1,s2,16).astype(int)[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Actual survival rate') 
    #plt.figtext(0.03,0.5,'%',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/actual_surv_rate_snapshot.png')
    plt.close()

# actual death rate series
def compute_actual_death_rate(s):
    """
    Takes age distribution of two consecutive stages. Computes the fraction 
    of those died from age x to age x+1.
    Returns a numpy array.
    """
    stage1 = age_distr_in[s]*pop_in[s]
    stage2 = np.array(list((age_distr_in[s+1]*pop_in[s+1]))[1:]+[0])
    div = stage1
    div[div == 0] = 1

#    print age_distr_in[s]*pop_in[s]
#    print age_distr_in[s+1]*pop_in[s+1]
#    print (stage1 - stage2) / div

    return (stage1 - stage2) / div

#def compute_actual_death_rate(s):
#    """
#    Computes the actual death rate as retrieved from age_distr_in.
#    Starts with generation born at stage s.
#    Returns a numpy array.
#    """
#    div = age_distr_in[s][0]*pop_in[s]
#    t = [0]*71
#    c = s+1
#    for age in range(1,71):
#        t[age] = (div - age_distr_in[c][age]*pop_in[c]) / div
#        c += 1
#        if t[age]==1:
#            break
#    return np.array(t)

def avr_actual_death_rate(s):
    """Averages actual death rate over 100 stages."""
    if s <= 50:
        res = compute_actual_death_rate(s)
        for i in range(s+1,s+101):
            res += compute_actual_death_rate(i)
        return res/100
    if s >= n_stage-120:
        res = compute_actual_death_rate(n_stage-70)
        for i in range(n_stage-170,n_stage-70):
            res += compute_actual_death_rate(i)
        return res/100
    else:
        res = compute_actual_death_rate(s+50)
        for i in range(s-50,s+50):
            res += compute_actual_death_rate(i)
        return res/100

def plot_actual_death_rate(s):
    """ Plots actual death rate averaged over 100 stages."""
    y = avr_actual_death_rate(s)
#    idx = list(t[1:]).index(0)
#    y = t[:idx]
    plt.plot(range(len(y)),y)
    plt.plot((16,16),(0,1.1),'r--')
    plt.axis([0,70,0,1.1])
    plt.title('Actual death rate')
    plt.xlabel('age')
    #plt.ylabel('N',rotation='horizontal')
    plt.text(60,1.125,'stage '+str(s))
    plt.savefig(file_out+'/actual_death_rate.png')
    plt.close()

def plot_log_actual_death_rate(s):
    """ Plots log10 of actual death rate averaged over 100 stages."""
    t = avr_actual_death_rate(s)
#    t = t[:(list(t[1:]).index(0))]
    t[t == 0] = 1
#    y = np.log10(t[1:(list(t[1:]).index(1))])
    y = np.log(t)
    x = range(len(y))
    # gompertz fitting
    coeff = np.polyfit(x,y,1)
    polynomial = np.poly1d(coeff)
    ys = polynomial(x)
    #print polynomial

    plt.scatter(x,y)
    plt.plot(x,ys,'k-')
    #plt.plot((16,16),(,,'r--')
    plt.title('log10 actual death rate')
    plt.xlabel('age')
    #plt.ylabel('N',rotation='horizontal')
    plt.figtext(0.8,0.95,'stage '+str(s))
    plt.savefig(file_out+'/log_actual_death_rate.png')
    plt.close()

def snapshot_actual_death_rate(s1=0,s2=n_stage):
    "Plot 4x4 snapshot plot of actual death rate, averaged over 100 stages."
    for i in range(1,17):
        y = avr_actual_death_rate(np.linspace(s1,s2,16).astype(int)[i-1])
        plt.subplot(4,4,i).scatter(range(71),y)
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,0.1])
        plt.tick_params(labelsize='small')
        plt.text(50,100.01,str(np.linspace(s1,s2,16).astype(int)[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Actual death rate') 
    #plt.figtext(0.03,0.5,'%',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/actual_death_rate_snapshot.png')
    plt.close()

def snapshot_log_actual_death_rate(s1=0,s2=n_stage):
    "Plot 4x4 snapshot plot of log10 of actual death rate, averaged over 100 stages."
    for i in range(1,17):
        y = np.log10(avr_actual_death_rate(np.linspace(s1,s2,16).astype(int)[i-1]))
        plt.subplot(4,4,i).plot(y,'k-')
        plt.plot((16,16),(0,1),'r--')
        plt.tick_params(labelsize='small')
        plt.text(50,100.01,str(np.linspace(s1,s2,16).astype(int)[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('log10 actual death rate') 
    #plt.figtext(0.03,0.5,'%',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/log_actual_death_rate_snapshot.png')
    plt.close()

## pop_pyramid_sex
def pop_pyramid_sex(s=n_stage):
    ages = [males_females_ages[0][:16]+[0]*55, males_females_ages[1][:16]+[0]*55, [0]*16+males_females_ages[0][16:], [0]*16+males_females_ages[1][16:]]
    types = ['Males juvenile', 'Females juvenile', 'Males adult', 'Females adult']
    pyramid_chart = pygal.Pyramid(human_readable=True, legend_at_bottom=True)
    pyramid_chart.title = 'Population by age and gender [stage '+str(s)+']'
    pyramid_chart.x_labels = map(lambda x: str(x) if not x % 5 else '', range(70))
    for t, age in zip(types, ages):
        pyramid_chart.add(t,age)
    pyramid_chart.render_to_file(file_out+'/pop_pyramid.svg')

## pop_pyramid_asex
def pop_pyramid_asex(s=n_stage):
    ages = [males_females_ages[:16]+[0]*55, [0]*16+males_females_ages[16:]]
#    types = ['Juvenile', 'Adult']
    horizontal_chart = pygal.HorizontalBar()
    horizontal_chart.title = 'Population by age [stage '+str(s)+']'
    horizontal_chart.x_labels = map(lambda x: str(x) if not x % 5 else '', range(70))
    horizontal_chart.add('Juvenile', ages[0])
    horizontal_chart.add('Adult', ages[1])
    horizontal_chart.render_to_file(file_out+'/pop_pyramid.svg')

## pop_res
def plot_pop_res(s1=0,s2=n_stage):
    "Plot the population and resources values from stage s1 to stage s2."
    l2,l1 = plt.plot(pop_in[s1:s2+1],'b-',res_in[s1:s2+1],'r-')
    plt.figure(1).legend((l1,l2),('resources','population'),'upper right',prop={'size':7})
    plt.title('Resources and population')
    plt.xlabel('stage')
    plt.ylabel('N',rotation='horizontal')
    plt.axis([s1,s2,0,max(max(res_in[s1:s2+1]),max(pop_in[s1:s2+1]))])
    plt.xticks(np.linspace(0,s2-s1,6),map(str,(np.linspace(s1,s2,6)).astype(int)))
    plt.savefig(file_out+'/pop_res.png')
    plt.close()

### SNAPSHOT
## age distribution 
def snapshot_age_distr(s1=0,s2=n_stage):
    """
    Plot 4x4 snapshot plot of age distribution.
    Plots 2-15 are averaged over +-5 stages.
    Plot 16 is averaged over -10 stages.
    """
    if s2-s1<200:
        print 'snapshot_age_distr: Range is too small. Plots overlap and are not reliable.'
        return
    if s1==0:
        plt.subplot(4,4,1).plot(age_distr_in[s1],'b-')
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,0.3])
        plt.text(50,0.301,str(s1),size=6)
        plt.tick_params(labelsize='small',labelbottom='off')
        var1 = 2
    else:
        var1 = 1
    if s2==n_stage:
        var = age_distr_in[s2]
        for i in range(-10,0,1):
            var += age_distr_in[-1+i]
        var = var/11
        plt.subplot(4,4,16).plot(var,'b-')
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,0.3])
        plt.text(50,0.301,str(s2),size=6)
        plt.tick_params(labelsize='small',labelleft='off')
        var2 = 16
    else:
        var2 = 17
    for i in range(var1,var2):
        var = np.zeros((71,))
        for k in range(-5,6,1):
            var += age_distr_in[np.linspace(s1,s2,16).astype(int)[i-1]+k]
        var = var/11
        plt.subplot(4,4,i).plot(var,'b-')
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,0.3])
        plt.tick_params(labelsize='small')
        plt.text(50,0.301,str(np.linspace(s1,s2,16).astype(int)[i-1]),size=6)
        if any(i==np.array([5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Age distribution') 
    plt.figtext(0.03,0.5,'fraction',rotation='vertical')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/age_distr.png')
    plt.close()

## survival rate
def snapshot_surv_rate():
    "Plot 4x4 snapshot plot of survival rate."
    for i in range(1,17):
        ipy = surv_rate_in[i-1]*100
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'b-')
        #sd
        ipy = surv_rate_sd_in[i-1]*100
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy_sd = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).fill_between(nipx,nipy+nipy_sd,nipy-nipy_sd,color='0.85')
        #junk
        ipy = np.array(list(surv_rate_junk_in[i-1]*100)*71)
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'g-')
        plt.plot((16,16),(98,100),'r--')
        plt.axis([0,70,down_limit_surv,100])
        plt.grid(color='k',linestyle='-')
        plt.tick_params(labelsize='small')
        plt.text(50,100.01,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Survival rate') 
    plt.figtext(0.03,0.5,'%',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/surv_rate.png')
    plt.close()

## cumulative survival rate
def snapshot_cum_surv_rate():
    "Plot 4x4 snapshot plot of cumulative survival rate."
    for i in range(1,17):
        ipy = np.array(surv_rate_in[i-1])
        for k in range(1,len(ipy)):
            ipy[k]=ipy[k-1]*ipy[k]
        ipy=ipy/ipy[0]
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'b-')
        #junk
        ipy = np.array(list(surv_rate_junk_in[i-1])*71)
        for k in range(1,len(ipy)):
            ipy[k]=ipy[k-1]*ipy[k]
        ipy=ipy/ipy[0]
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'0.5')
        plt.plot((16,16),(0,1),'r--')
        plt.axis([0,70,0,1])
        plt.grid(color='k',linestyle='-')
        plt.tick_params(labelsize='small')
        plt.text(50,1.01,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Normalized cumulative survival rate') 
    plt.figtext(0.03,0.5,'survival rate',rotation='vertical')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/cum_surv_rate.png')
    plt.close()

## reproduction rate
def snapshot_repr_rate():
    "Plot 4x4 snapshot plot of reproduction rate."
    for i in range(1,17):
        ipy = np.array(repr_rate_in[i-1]*100)
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'b-')
        #sd
        ipy = np.array(repr_rate_sd_in[i-1]*100)
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy_sd = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).fill_between(nipx,nipy+nipy_sd,nipy-nipy_sd,color='0.85')
        #junk
        ipy = np.array(list(repr_rate_junk_in[i-1]*100)*71)
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'g-')
        plt.axis([16,70,0,up_limit_repr])
        plt.grid(color='k',linestyle='-')
        plt.tick_params(labelsize='small')
        plt.text(50,up_limit_repr+0.01,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Reproduction rate') 
    plt.figtext(0.03,0.5,'%',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/repr_rate.png')
    plt.close()

## fitness
#def snapshot_fit():
#    "Plot 4x4 snapshot plot of fitness."
#    for i in range(1,17):
#        ipy = np.array(fit_in[i-1])
#        tck = interpolate.splrep(ipx,ipy,s=0)
#        nipy = interpolate.splev(nipx,tck,der=0)
#        plt.subplot(4,4,i).plot(nipx,nipy,'b-')
#        #junk
#        ipy = np.array(list(fit_junk_in[i-1])*71)
#        tck = interpolate.splrep(ipx,ipy,s=0)
#        nipy = interpolate.splev(nipx,tck,der=0)
#        plt.subplot(4,4,i).plot(nipx,nipy,'g-')
#        plt.axis([16,70,0,1])
#        plt.grid(color='k',linestyle='-')
#        plt.tick_params(labelsize='small')
#        plt.text(50,1.01,str(snapshot_stages[i-1]),size=6)
#        if any(i==np.array([1,5,9])):
#            plt.tick_params(labelbottom='off')
#        elif any(i==np.array([14,15,16])):
#            plt.tick_params(labelleft='off')
#        elif i!=13:
#            plt.tick_params(labelbottom='off',labelleft='off')
#    plt.suptitle('Fitness') 
#    plt.figtext(0.03,0.5,'fit',rotation='horizontal')
#    plt.figtext(0.5,0.03,'age')
#    plt.savefig(file_out+'/fit.png')
#    plt.close()

## fitness
def compute_fit(idx):
    """Fitness calculated as Dario defined it."""
    survv = np.array(surv_fit_in[idx])
    reprv = np.array(repr_fit_in[idx])
    k = np.prod(survv[:16])
    t = [0]*16
    for i in range(16,71):
        t.append(reprv[i] * np.prod(survv[16:i+1]))
    return k * np.array(t)

def plot_sum_fit():
    """Plot the sum for fitness."""
    t = []
    for i in range(16):
        t.append(np.sum(compute_fit(i)))
    plt.scatter(snapshot_stages,t)
    plt.plot(snapshot_stages,t)
    plt.axis([0,n_stage,0,max(t)])
    plt.title('Sum relative fitness')
    plt.xlabel('stage')
    plt.ylabel('sum(F)',rotation='horizontal')
    plt.savefig(file_out+'/sum_fit.png')
    plt.close()

def snapshot_fit():
    "Plot 4x4 snapshot plot of fitness."
    ymax = max(compute_fit(0))
    for i in range(1,16):
        if ymax < max(compute_fit(i)):
            ymax = max(compute_fit(i))
    for i in range(1,17):
        ipy = compute_fit(i-1)
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'b-')
        plt.axis([16,70,0,ymax])
        plt.grid(color='k',linestyle='-')
        plt.tick_params(labelsize='small')
        plt.text(50,ymax+0.0025,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Relative fitness') 
    plt.figtext(0.03,0.5,'F',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/fit.png')
    plt.close()
    
## survival-reproduction difference
def snapshot_surv_repr_diff():
    "Plot 4x4 snapshot plot of survival-reproduction difference."
    for i in range(1,17):
        ipy = np.array(surv_fit_in[i-1]-repr_fit_in[i-1])
        tck = interpolate.splrep(ipx,ipy,s=0)
        nipy = interpolate.splev(nipx,tck,der=0)
        plt.subplot(4,4,i).plot(nipx,nipy,'r-')
        plt.plot((0,70),(0,0),'k-')
        plt.axis([16,70,-0.5,0.5])
        plt.tick_params(labelsize='small')
        plt.text(50,0.501,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('Survival-reproduction difference') 
    plt.figtext(0.03,0.5,r'$\Delta$',rotation='horizontal')
    plt.figtext(0.5,0.03,'age')
    plt.savefig(file_out+'/surv_repr_diff.png')
    plt.close()

## density
def snapshot_dens():
    "Plot 4x4 snapshot density plot (distribution of 1's)."
    for i in range(1,17):
        l1,l2 = plt.subplot(4,4,i).plot(dens_surv_in[i-1],'g-',dens_repr_in[i-1],'r-')
        plt.axis([0,20,0,1])
        plt.tick_params(labelsize='small')
        plt.text(15,1.01,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.figlegend((l1,l2),('survival','reproduction'),'upper right',prop={'size':7})
    plt.suptitle('Density of genetic units') 
    plt.figtext(0.03,0.5,'weight',rotation='vertical')
    plt.figtext(0.5,0.03,'genetic unit')
    plt.savefig(file_out+'/dens.png')
    plt.close()

## heterozygosity measure
def snapshot_hetrz_mea():
    "Plot 4x4 snapshot plot of heterozygosity measure."
    for i in range(1,17):
        plt.subplot(4,4,i).scatter(range(1260),hetrz_mea[i-1],s=5,c='b',marker='.')
        plt.plot((710,710),(0,1),'r-')
        plt.plot((170,170),(0,1),'r--')
        plt.axis([0,1260,0,1])
        plt.yticks([0,1])
        plt.xticks([1,711,1260])
        plt.tick_params(labelsize='small')
        plt.text(1000,1.01,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('average frequency of 1\'s per locus') 
    plt.figtext(0.03,0.5,r'$\bar{f}$',rotation='horizontal')
    plt.figtext(0.5,0.03,'locus')
    plt.savefig(file_out+'/hetrz_mea_snapshot.png')
    plt.close()

## age-dependant genotypical diversity measure
def snapshot_hetrz_mea_sd():
    "Plot 4x4 snapshot plot of standard deviation for heterozygosity measure."
    ymax = max(hetrz_mea_sd[0])
    for i in range(1,16):
        if ymax < max(hetrz_mea_sd[i]):
            ymax = max(hetrz_mea_sd[i])

    ymin = min(hetrz_mea_sd[0])
    for i in range(1,16):
        if ymin > min(hetrz_mea_sd[i]):
            ymin = min(hetrz_mea_sd[i])

    x = range(1260)
    for i in range(1,17):
        y = hetrz_mea_sd[i-1]
        # gompertz fitting
        coeff = np.polyfit(x[:711],y[:711],6)
        polynomial = np.poly1d(coeff)
        ys = polynomial(x)

        coeff = np.polyfit(x[711:],y[711:],6)
        polynomial = np.poly1d(coeff)
        yr = polynomial(x)

        #plt.subplot(4,4,i).scatter(x,y,s=5,c='b',marker='.')
        plt.subplot(4,4,i).plot(x[:711],ys[:711],'b-')
        plt.subplot(4,4,i).plot(x[711:],yr[711:],'b-')
        plt.plot((710,710),(ymin,ymax),'r-')
        plt.plot((170,170),(ymin,ymax),'r--')
        plt.axis([0,1260,ymin,ymax])
#        plt.yticks([0,1])
        plt.xticks([1,711,1260])
        plt.tick_params(labelsize='small')
        plt.text(1000,ymax+0.0025,str(snapshot_stages[i-1]),size=6)
        if any(i==np.array([1,5,9])):
            plt.tick_params(labelbottom='off')
        elif any(i==np.array([14,15,16])):
            plt.tick_params(labelleft='off')
        elif i!=13:
            plt.tick_params(labelbottom='off',labelleft='off')
    plt.suptitle('frequency of 1\'s per locus-sd') 
    plt.figtext(0.03,0.5,'sd',rotation='horizontal')
    plt.figtext(0.5,0.03,'locus')
    plt.savefig(file_out+'/hetrz_mea_sd_snapshot.png')
    plt.close()

## plot all snapshots
def snapshot_all(s1=0,s2=n_stage):
    "Plot all snapshot plots."
    snapshot_age_distr()
    snapshot_surv_rate()
    snapshot_cum_surv_rate()
    snapshot_repr_rate()
    snapshot_fit()
    snapshot_surv_repr_diff()
    snapshot_dens()
    snapshot_hetrz_mea()
    snapshot_hetrz_mea_sd()

### average over range/single plots
## age distribution
def avrg_age_distr(s1,s2=0):
    "Plot age distribution averaged over stages s1-s2."
    if s2==0:
        s2=s1
        var2=1
    else:
        var2=s2-s1+1
    var = np.zeros((71,))
    for i in range(s1,s2+1):
        var += np.array(age_distr_in[i])
    var = var/var2
    plt.plot(var,'b-')
    plt.title('Age distribution')
    plt.xlabel('age')
    plt.ylabel('fraction',rotation='vertical')
    plt.axis([0,70,0,0.3])
    if s2!=s1:
        plt.text(50,0.26,'stage '+str(s1)+'-'+str(s2))
    else:
        plt.text(50,0.26,'stage '+str(s1))
    plt.savefig(file_out+'/avrg_age_distr.png')
    plt.close()

### timelapse
## age distribution
def timelapse_age_distr(s1,s2,t):
    "Age distribution timelapse stage s1-s2"
    plt.ion()
    plt.axis([0,70,0,0.3])
    plt.title('Age distribution')
    plt.ylabel('fraction',rotation='vertical')
    plt.xlabel('age')
    fig_txt = plt.figtext(0.8,0.95,'stage '+str(s1))
    fig_txt.set_bbox(dict(facecolor='red', alpha=0.5))
    plt.draw()
    for i in range(s1,s2):
        var = plt.plot(age_distr_in[i],'b-')
        fig_txt.set_text('stage '+str(i))
        plt.draw()
        sleep(t)
        var[0].remove()
    plt.ioff()
    plt.plot(age_distr_in[s2],'b-')
    fig_txt.set_text('stage '+str(s2))
    sleep(10)

## code to be executed
pop_pyramid_asex()
plot_pop_res()
snapshot_all()
plot_sum_fit()

#snapshot_actual_surv_rate()
#snapshot_actual_death_rate()
plot_actual_surv_rate(200)
plot_actual_death_rate(200)
plot_log_actual_death_rate(200)
