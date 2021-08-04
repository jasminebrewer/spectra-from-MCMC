#!/usr/bin/env python3

import emcee
from scipy.optimize import minimize
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import corner
import ipdb

#########################################
# Data processing #######################
#########################################    

[ptc,ptlo,pthi,xs,stat,sys,lumi] = range(0,7)
sigmaNN = [61.8, 67.6, 70.9] # mb, for 2.76, 5.02, and 7 TeV respectively (1710.07098)

def process_hadrons(sys_err):
    
    all_filenames = ["ALICE_2.76TeV_hadron_spectra_new.csv","ALICE_5.02TeV_hadron_spectra.csv","ALICE_7TeV_hadron_spectra.csv"]
    all_roots = [2760, 5020, 7000]
    all_data = [np.loadtxt('data/'+filename) for filename in all_filenames]
    [ptc,ptlo,pthi,xs,statp,statm,sysp,sysm,lumip,lumim] = range(0,10) # indices for various quantities in the data, plus two extra luminosity indices to add
   
    # add luminosity indices
    all_data = [np.hstack( (all_data[nc],np.zeros( (len(all_data[nc]),2) ) ) ) for nc in range(0,len(all_data))]
    
    lumi_error_percentages = [0.019,0.023,0.036]

    for nc in range(0,len(all_data)):
        all_data[nc][:,lumip] = all_data[nc][:,xs]*lumi_error_percentages[nc]
        all_data[nc][:,lumim] = -all_data[nc][:,xs]*lumi_error_percentages[nc]

    # Convert from N to sigma in 2.76 and 5.02 TeV data
    need_correction_indices = [0,1]
    for nci, nc in enumerate(need_correction_indices):
        all_data[nc][:,xs:(lumim+1)] = np.transpose(np.array([all_data[nc][:,i]*sigmaNN[nci] for i in range(xs,(lumim+1))]))
        
    # multiply 7TeV spectra by 2*pi*pT to obtain dsigma/dpT
    all_data[2][:,xs:(lumim+1)] = np.transpose(np.array([all_data[2][:,i]*(2*np.pi*all_data[2][:,ptc]) for i in range(xs,(lumim+1))]))
    
    # [ptc,ptlo,pthi,xs,stat,sys,lumi] = range(0,7)
    
    stat_indices = [[statp,statm]]
    sys_indices = [[sysp,sysm]]
    lumi_indices = [[lumip,lumim]]
    all_data = [consolidate_uncertainties( data,stat_indices,sys_indices,lumi_indices ) for data in all_data]
           
    return [all_roots, all_data, all_filenames]


def process_CMS_hadrons(sys_err):
    
    all_filenames = ["CMS_2.76TeV_hadron_spectra.csv","CMS_5.02TeV_hadron_spectra.csv","CMS_7TeV_hadron_spectra.csv"]
    all_roots = [2760, 5020, 7000]
    file_delimiters = ['\t',',','\t']
    all_data = [np.loadtxt('data/'+filename,delimiter=file_delimiters[i]) for i,filename in enumerate(all_filenames)]
    [ptc,ptlo,pthi,xs,statp,statm,sysp,sysm,lumip,lumim] = range(0,10) # indices for various quantities in the data, plus two extra luminosity indices to add
   
    # add luminosity indices
    all_data = [np.hstack( (all_data[nc],np.zeros( (len(all_data[nc]),2) ) ) ) for nc in range(0,len(all_data))]
    
    lumi_error_percentages = [0.06,0.023,0.04]
    for nc in range(0,len(all_data)):
        all_data[nc][:,lumip] = all_data[nc][:,xs]*lumi_error_percentages[nc]
        all_data[nc][:,lumim] = -all_data[nc][:,xs]*lumi_error_percentages[nc]
        
    # Convert from N to sigma
    need_correction_indices = [0,1,2]
    for nci, nc in enumerate(need_correction_indices):
        all_data[nc][:,xs:(lumim+1)] = np.transpose(np.array([all_data[nc][:,i]*sigmaNN[nci] for i in range(xs,(lumim+1))]))

    # multiply all spectra by 2*pi*pT to obtain d^2N/dpT dy
    for n in range(0,len(all_data)):
        all_data[n][:,xs:(lumim+1)] = np.transpose(np.array([all_data[n][:,i]*(2*np.pi*all_data[n][:,ptc]) for i in range(xs,(lumim+1))]))
    
    # multiply 7TeV by correction factor
    all_data[2][:,xs:(lumim+1)] = np.transpose(np.array([all_data[2][:,i]*0.9705 for i in range(xs,(lumim+1))]))
        
    stat_indices = [[statp,statm]]
    sys_indices = [[sysp,sysm]]
    lumi_indices = [[lumip,lumim]]
    all_data = [consolidate_uncertainties( data,stat_indices,sys_indices,lumi_indices ) for data in all_data]
               
    return [all_roots, all_data, all_filenames]
    

def process_jets(sys_err):
    
    filename276='data/ATLAS_2.76TeV_jet_spectra.csv'
    file276=np.genfromtxt(filename276, delimiter=',', skip_footer=10)
    filename7='data/ATLAS_7TeV_jet_spectra.csv'
    file7=np.genfromtxt(filename7, delimiter=',')
    [ptc,ptlo,pthi,xs,statp,statm,sys1p,sys1m,sys2p,sys2m,sys3p,sys3m] = range(0,12) # indices for various quantities in the data
    # [ptc,ptlo,pthi,xs,stat,sys,lumi] = range(0,6)

    stat_indices = [[statp,statm]]
    sys_indices = [[sys1p,sys1m],[sys2p,sys2m]]
    lumi_indices = [[sys3p,sys3m]]
    file276 = consolidate_uncertainties( file276,stat_indices,sys_indices,lumi_indices )
    file7 = consolidate_uncertainties( file7,stat_indices,sys_indices,lumi_indices )
    
    # convert 7 TeV from pb to nb
    file7[:,xs:] = file7[:,xs:]/(10**3)

    filename5='data/ATLAS_5.02TeV_jet_spectra.csv'
    file5=np.genfromtxt(filename5, delimiter=',')

    # this file has systematic instead of statistical uncertainties first, so interchange those columns...
    file5 = np.array([file5[:,i] for i in [ptc,ptlo,pthi,xs,sys1p,sys1m,statp,statm,sys2p,sys2m]]).T

    stat_indices = [[statp,statm]] 
    sys_indices = [[sys1p,sys1m]]
    lumi_indices = [[sys2p,sys2m]]
    file5 = consolidate_uncertainties( file5,stat_indices,sys_indices,lumi_indices )

    all_roots = [2760, 5020, 7000]
    all_filenames = [filename276, filename5, filename7]
    all_data = [file276, file5, file7]
    
    return [all_roots, all_data, all_filenames]


# function to sum errors in index_pairs in quadrature
def quad_sum(data, index_pairs):
    
    return np.array([np.sqrt(np.sum([((datum[pair[0]]-datum[pair[1]])/2)**2 for pair in index_pairs])) for datum in data])


def consolidate_uncertainties( thisfile, stat_indices, sys_indices, lumi_indices ):
    
    if sys_err=='lumi_only':
        
        quad_sys = np.zeros( len(thisfile[:,stat_indices[0]]) )
        
    elif sys_err=='uncorr': 
        
        quad_sys = quad_sum( thisfile, sys_indices )
        
    stat_err = quad_sum( thisfile, stat_indices )
    lumi_err = quad_sum( thisfile, lumi_indices )
    
    consolidated = np.array([thisfile[:,i] for i in [ptc,ptlo,pthi,xs,stat,sys,lumi]]).T
    
    for ind,err in zip( [stat,sys,lumi],[stat_err, quad_sys, lumi_err] ):
        
        consolidated[:,ind] = err
        
    return consolidated

#########################################
# Error matrices ########################
#########################################

def construct_error_mat(data):
        
    total_length = np.sum([len(datum) for datum in data])
    lumimat = np.zeros( (total_length,total_length) )
    
    lengths = [0]+[len(datum) for datum in data]
    cumulative = [np.sum(lengths[:(i+1)]) for i in range(len(lengths))]
    for ind in range(0,len(cumulative)-1):
        
        lumi_err = data[ind][:,lumi]
        ind1 = cumulative[ind]
        ind2 = cumulative[ind+1]
        lumimat[ ind1:ind2,ind1:ind2 ] = [[lumi_err[i]*lumi_err[j] for i in range(len(lumi_err))] for j in range(len(lumi_err))]
    
    statmat = np.diag( np.concatenate([datum[:,stat]**2 for datum in data]) )
    sysmat = np.diag( np.concatenate([datum[:,sys]**2 for datum in data]) )
    errmat = statmat + sysmat + lumimat
        
    return np.linalg.inv( errmat )

    

#########################################
# General helper functions ##############
#########################################

def xT(pT, sqrts):
    
    return 2*pT/sqrts    

# Fast integration method for binning fits
def gauss_legendre_integral(model_func,x,theta):
    
    xc = x[:,0]
    dx = x[:,2]-x[:,1]
    xvals = [xc-np.sqrt(3/5)*dx/2,xc,xc+np.sqrt(3/5)*dx/2]
    return (5/18)*model_func(*theta,xvals[0]) + (8/18)*model_func(*theta,xvals[1]) + (5/18)*model_func(*theta,xvals[2])


def int_model(model_func,x,theta):
    
    return gauss_legendre_integral(model_func,x,theta)


def mask_data(data, minpT):
    
    masks = [ data[i][:,ptc] > minpT for i,_ in enumerate(data)]
    return [data[i][ masks[i] ] for i,_ in enumerate(data)]


def make_dir(name, lines):
    
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)
    file = open("README.txt","w+")
    file.writelines(lines)
    file.close()



#########################################
# Maximum likelihood estimation #########
# (used for starting point of MCMC) #####
#########################################

def do_MLE(data, error_matrix, roots, model_func, initial_point, ntimes=1, debug_mode=False):

    if debug_mode:
        
        plt.plot(data[0][:,ptc],data[0][:,xs],'b', label='data')
        plt.plot(data[0][:,ptc],model_func(*initial_point,roots[0],xT(data[0][:,ptc],roots[0])),'r', label='fit')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()
        ipdb.set_trace()

    nll = lambda *args: -joint_likelihood(*args)
    if ntimes==1:
        soln = minimize(nll, initial_point, args=(roots, data, error_matrix, model_func))
    else:
        # keep in mind that 0s in initial point don't get varied. Consider changing that.
        min_fun = np.inf
        for n in range(0,ntimes):
            new_initial_point = random.uniform(0.9*initial_point,1.1*initial_point)
            new_soln = minimize(nll, initial_point, args=(roots, data, error_matrix, model_func))
            if new_soln.fun < min_fun:
                min_soln = new_soln
                min_fun = new_soln.fun
        soln = min_soln
    return [soln.x, 2*soln.fun]
    

def plot_mle(data, roots, model_func, theta, chisqr):
    
    fig, ax = plt.subplots()
    for i,datum in enumerate(data):
        
        xnow = xT(datum[:,ptc:(pthi+1)],roots[i])
        ax.errorbar( xnow[:,0],datum[:,xs],datum[:,stat], color=colors[i], capsize=5, fmt='o')
        ax.fill_between( xnow[:,0],datum[:,xs]-datum[:,sys]/2,datum[:,xs]+datum[:,sys]/2, color='grey', alpha=0.3)
        ax.plot( xnow[:,0], int_model(model_func, xnow,np.append(theta,roots[i])), color='k', linestyle='--')
        
    ax.set_title(r'$\chi^2=$'+str(chisqr))
    ax.set_yscale('log')
    ax.set_xscale('log')
        
    fig.savefig('mle.png')


#########################################
# MCMC ##################################
#########################################

######### Helper functions #############
def log_prior(theta, bounds):

    if in_bounds( theta, bounds ):
        return 0.0
    else:
        return -np.inf

    
def log_probability(theta, sqrts, data, error_matrix, model_func, bounds):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + joint_likelihood(theta, sqrts, data, error_matrix, model_func)


# helper function for joint_likelihood    
def delta_y(theta, x, y, model_func):
    
    model = int_model(model_func, x, theta)
    return (y-model)
    

def joint_likelihood(theta, sqrts, data, error_matrix, model_func):
    
    all_deltay = np.concatenate( [delta_y( np.append(theta,sqrts[i]), xT(datum[:,ptc:(pthi+1)],sqrts[i]), datum[:,xs], model_func) for i,datum in enumerate(data)])

    return -0.5 * np.dot( all_deltay, np.dot(error_matrix, all_deltay) )


def in_bounds(theta, bounds):
    
    params_in_bounds = [min(bounds[i])<=theta[i]<=max(bounds[i]) for i in range(len(theta))]
    
    return np.all(params_in_bounds)
 
##### Main function to run MCMC ########
def do_MCMC(data, error_matrix, roots, model_func, ndim, nwalkers, nsamples, theta_start, bounds, pos_radius):
    
    pos = [theta_start*(1 + pos_radius * np.random.randn(ndim)) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(roots, data, error_matrix, model_func, bounds))
    sampler.run_mcmc(pos, nsamples, progress=True)
    return sampler

def make_samples(masked_data, roots, model_func, minpT, initial, bounds, sys_err, pos_radius=1e-1):
    
    if len(initial) != ndim:
        print('length of initial condition should match number of parameters')
    
    error_matrix = construct_error_mat(masked_data)
    
    [theta_mle, chisqr_mle] = do_MLE(masked_data, error_matrix, roots, model_func, initial, 100, debug_mode=False)
    plot_mle( masked_data, roots, model_func, theta_mle, chisqr_mle)
    sampler = do_MCMC( masked_data, error_matrix, roots, model_func, ndim, 200, 10000, theta_mle, bounds, pos_radius)
    flat_samples = plot_sampler(sampler, 8000,20,param_labels)

    return [flat_samples, theta_mle] 


# Plot results of the MCMC ##############
def plot_sampler(sampler, burn_in, thin, labels):
    
    ndim = len(labels)
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("step number")
    fig.savefig('samples.png')
    
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    fig = corner.corner(flat_samples)
    fig.savefig('corner.png')
    
    return flat_samples
    

##########################################
# Fitting models #########################
##########################################

# 4-parameter model for fitting individual energies
def model_individ(alpha, a, b,c, sqrts, x):

    return ((alpha/(sqrts/2760))**3)*(x/x0)**(a+b*np.log(x/x0)+c*x/x0)

# 8-parameter models for global fits of all energies

def model_log(sqrts0, beta, a, b, c, d, e, f, sqrts, x):

    return ((sqrts/sqrts0)**beta)*(x/x0)**(a+ c*(x/x0) + e*np.log(x/x0) +np.log(sqrts/2760)*( d + b*(x/x0) + f*np.log(x/x0) ) )



def model_lin(sqrts0, beta, a, b, c, d, e, f, sqrts, x):

    return ((sqrts/sqrts0)**beta)*(x/x0)**(a+ c*(x/x0) + e*np.log(x/x0) +(sqrts/2760)*( d + b*(x/x0) + f*np.log(x/x0) ) )


# Run the fitting
def do_fits(model_type, data_indices, flag=''):
    
    # make some aesthetic/ practical stuff global so we don't have to pass it everywhere
    global ndim, savedir, model_string, colors, filenames, param_labels

    data = [all_data[ di ] for di in data_indices]
    roots = [ all_roots[ di ] for di in data_indices]
    filenames = [ all_filenames[ di ] for di in data_indices]
    colors = [all_colors[ di ] for di in data_indices]
    
    masked_data = mask_data( data,minpT )
    
    if system=='hadrons':
        initial_size=1e-3
    if system=='jets':
        initial_size=1e-2    
                
    if model_type=='individ':
        
        ndim=4
        model_string = 'alpha*(x/x0)**(a+b*np.log(x/x0)+c*x/x0)'
        param_labels = ["alpha","a", "b","c"]
        
        if system=='hadrons' or system=='CMShadrons':
            # initial_point = np.array([0.1,-4.9,0,0])
            initial_point = np.array([0.25,-5.2,0,0])
            #initial_point = np.array([0.01,-5.2,0,0])
        if system=='jets':
            #initial_point = np.array([10**3,-5.9,0,0])
            initial_point = np.array([10,-5.9,0,0])
            
        mcmc_bounds = np.array([[0,np.inf],[-np.inf,0],[-np.inf,np.inf],[-np.inf,np.inf]])
        
        
    elif model_type=='lin' or model_type=='log':

        ndim=8
        param_labels = ["sqrts0","beta","a", "b","c","d","e","f"]
        
        if system=='hadrons' or system=='CMShadrons':
            initial_point = np.array([10**3,-3.9,-5,0,0,0,0,0])
            #initial_point = np.array([1500.,-4.3,-5,0,0,0,0,0])
            #initial_point = np.array([10**3.1,-3.5,-5,0,0,0,0,0])
        if system=='jets':
            initial_point = np.array([10**4,-3.9,-5,0,0,0,0,0])
            
        mcmc_bounds = np.array([[0,np.inf],[-np.inf,0],[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf],[-np.inf,np.inf]])       
        
        if model_type=='lin':
            model_string = '(sqrts/sqrts0)**beta)*(x/x0)**(a+c*(x/x0)+e*log(x/x0) +(sqrts/2760)*( d +b*(x/x0) + f*log(x/x0))'
        if model_type=='log':
            model_string = '(sqrts/sqrts0)**beta)*(x/x0)**(a+c*(x/x0)+e*log(x/x0) +log(sqrts)*( d +b*(x/x0) + f*log(x/x0))'
                    
    else:
        print('allowed values of "model" are "individ", "lin", and "log" and none of those were supplied.')
        return -1
    
    if not os.path.exists('output'):
        os.mkdir('output')
    os.chdir('output')
        
    savedir = system+index_string+'_'+str(minpT)+'_ndim'+str(ndim)+'_'+model_type+'_'+sys_err
    lines = [model_string+'\n','pTmin='+str(minpT)+'\n']+[filename+'\n' for filename in filenames]
    energy_labels=[str(rs)+' GeV' for rs in roots]
    make_dir(savedir, lines)
    
    if model_type=='individ':
        np.savetxt('data.txt',masked_data[0])
        [sample, theta_mle] = make_samples(masked_data, roots, model_individ, minpT, initial_point, mcmc_bounds, sys_err, initial_size)
        
        for i,d in enumerate(masked_data):
            get_spectra_percentiles( model_individ, sample, theta_mle, d[:,ptc:(pthi+1)], roots[i] )
    
    if system=='hadrons' or system=='CMShadrons':
        pTrange = np.linspace( 7,50,100 )
    if system=='jets':
        pTrange = np.linspace( 40,200,100 )

    if model_type=='lin':
        [sample, theta_mle] = make_samples(masked_data, roots, model_lin, minpT, initial_point, mcmc_bounds, sys_err, initial_size)      

        for i,d in enumerate(masked_data):
            get_spectra_percentiles( model_lin, sample, theta_mle, d[:,ptc:(pthi+1)], roots[i] )

        get_ratio( model_lin, sample, theta_mle, pTrange, 6370, 5020)
    
    if model_type=='log':
        [sample, theta_mle] = make_samples(masked_data, roots, model_log, minpT, initial_point, mcmc_bounds, sys_err, initial_size)
    
        for i,d in enumerate(masked_data):
            get_spectra_percentiles( model_log, sample, theta_mle, d[:,ptc:(pthi+1)], roots[i] )

        get_ratio( model_log, sample, theta_mle, pTrange, 6370, 5020)
        
    os.chdir('../..')
            
            
def save_spectra(sample, model_func,):
    
    for i,d in enumerate(masked_data):
        get_spectra_percentiles( model_log, sample, theta_mle, d[:,ptc:(pthi+1)], roots[i] )
    if system=='hadrons':
        pTrange = np.linspace( 7,50,100 )
    if system=='jets':
        pTrange = np.linspace( 40,200,100 )
    get_ratio( model_func, sample, theta_mle, pTrange, 6370, 5020)
    

def get_spectra_percentiles(model_func, sample, theta_mle, pTrange, sqrts):
    
    # **binned** (integrated) spectra for comparison to measurements
    model = np.array([int_model(model_func, xT(pTrange, sqrts),np.append(thetanow,sqrts)) for thetanow in sample])
    model_mle = int_model(model_func, xT(pTrange, sqrts),np.append(theta_mle,sqrts))
    percentiles = np.array([np.concatenate( (np.append( pTrange[i], model_mle[i] ), np.percentile(model[:,i],[10, 16, 50, 84, 90])) ) for i,_ in enumerate(pTrange)])
    np.savetxt('spectra_'+str(sqrts)+'.txt', percentiles)

def get_ratio(model_func, sample, theta_mle, pTrange, sqrts1, sqrts2):
    
    # **unbinned** (unintegrated) spectra ratio
    model = np.array([model_func(*thetanow,sqrts1,xT(pTrange, sqrts1))/model_func(*thetanow,sqrts2,xT(pTrange, sqrts2)) for thetanow in sample])
    
    percentiles = np.array([np.concatenate( ( np.array( [pTrange[i], model_func(*theta_mle,sqrts1,xT(pTrange[i], sqrts1))/model_func(*theta_mle,sqrts2,xT(pTrange[i], sqrts2))] ), np.percentile(model[:,i],[10, 16, 50, 84, 90])) ) for i,_ in enumerate(pTrange)])
    
    np.savetxt('ratio_'+str(sqrts1)+'over'+str(sqrts2)+'.txt', percentiles)
    

    
#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':

    systems=['hadrons','jets','CMShadrons']
    sys_errs=['lumi_only','uncorr']
    all_colors = ['magenta', 'green', 'orange']

    for system in systems:

        for sys_err in sys_errs:

            if not (system=='hadrons' or system=='jets' or system=='CMShadrons'):
                print('allowed values of system are "hadrons", "CMShadrons" and "jets" and something else was supplied. Proceeding with hadrons.')
                system='hadrons'
            if not (sys_err=='lumi_only' or sys_err=='uncorr'):
                print('allowed values of sys_err are "lumi_only" and "uncorr" and something else was supplied. Proceeding assuming uncorrelated systematic error.')
                sys_err = 'uncorr'

            if system=='hadrons':
                x0=0.003
                minpT = 7
                [all_roots, all_data, all_filenames] = process_hadrons(sys_err)

            if system=='CMShadrons':
                x0=0.003
                minpT = 7
                [all_roots, all_data, all_filenames] = process_CMS_hadrons(sys_err)
                

            if system=='jets':
                x0=0.02
                minpT = 40
                [all_roots, all_data, all_filenames] = process_jets(sys_err)

            [ptc,ptlo,pthi,xs,statp,statm,sysp,sysm] = range(0,8)

            model_type = 'individ'
            datasets = [[0],[1],[2]]

            for data_indices in datasets:

                index_string = ''.join([str(di) for di in data_indices])
                do_fits(model_type, data_indices)



            datasets = [[0,1,2],[1,2]]
            model_types = ['log','lin']

            for data_indices in datasets:

                for model_type in model_types:

                    index_string = ''.join([str(di) for di in data_indices])
                    do_fits(model_type, data_indices)

