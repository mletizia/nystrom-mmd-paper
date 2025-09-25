import numpy as np
from utils import power_interval

import matplotlib.pyplot as plt

EPS = 1e-5

def plot_powervsvars(results, vars, config, nys_feat=4, ctt_feat=4, legend_loc='upper left', file=False, ylim=None, xlim=None):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'ctt': rf'CTT  ($g={ctt_feat}$)',
                'fullrank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'gray',
                'fullrank': '#984ea3'
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'fullrank': 'd'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    # compute power for each method
    powers = {}
    for method in methods:
        if method=='fullrank':
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1]])

            nvar_fullrank = results[method].shape[0]

            print(nvar_fullrank)

            plt.plot(vars[:nvar_fullrank], powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
            plt.fill_between(vars[:nvar_fullrank], 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
            
        else: 
            if method=='ctt': n_feat=ctt_feat
            else: n_feat=nys_feat
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

            plt.plot(vars, powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
            plt.fill_between(vars, 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
    
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
        
    plt.ylabel(r'Power', fontsize=26)
    plt.xlabel(xlabel_dict[dataset], fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.2), ncol=2, fontsize=24)
    #plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.tight_layout()
    plt.grid()
    if file: plt.savefig(file)
    plt.show()


def plot_powervscomp(results, var, config, legend_loc='upper left', file=False, xlim=None, xlog=True, ylog=False):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours)',
                'rlss': r'Nyström-AKRLS (ours)',
                'rff': r'RFF',
                'ctt': r'CTT',
                'fullrank': 'Exact MMD',
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'fullrank': 'd'
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'grey',
                'fullrank': '#984ea3'
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    # compute power for each method
    powers_time = {}
    for method in methods:
        powers_time[method] = np.asarray([power_interval(el, niter) for el in results[method][var,1, :]])

        plt.plot(results[method][var,0,:], powers_time[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
        plt.fill_between(results[method][var,0,:], 
                 powers_time[method][:,1], 
                 powers_time[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    if xlim: plt.xlim(xlim)

    plt.ylabel(r'Power', fontsize=26)
    plt.xlabel('Computation time (s)', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.2), ncol=2, fontsize=24)
    #plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.tight_layout()
    if ylog==True: plt.yscale('log')
    if xlog==True: plt.xscale('log')
    plt.grid()
    if file: plt.savefig(file)
    plt.show()


def plot_powervsnfeat(results, var, config, legend_loc='upper left', file=False, ignore_ctt=True):
    # plot power vs number of random features

    label_dict = {
                'uniform': r'Nyström-uniform (ours)',
                'rlss': r'Nyström-AKRLS (ours)',
                'rff': r'RFF',
                'ctt': r'CTT',
                'full_rank': 'Exact MMD',
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'full_rank': 'd'
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'grey',
                'full_rank': '#984ea3'
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    # compute power for each method
    powers_time = {}
    if ignore_ctt: methods = [item for item in methods if item != "ctt"]
    for method in methods:
        powers_time[method] = np.asarray([power_interval(el, niter) for el in results[method][var,1, :]])

        plt.plot(results[method][var,2,:], powers_time[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
        plt.fill_between(results[method][var,2,:], 
                 powers_time[method][:,1], 
                 powers_time[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    plt.ylabel(r'Power', fontsize=26)
    plt.xlabel(r'$\ell$', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.2), ncol=2, fontsize=24)
    plt.tight_layout()
    plt.xscale('log')
    plt.grid()
    if file: plt.savefig(file)
    plt.show()


def plot_sizevsvars(results, vars, config, nys_feat=4, ctt_feat=3, legend_loc='upper left', file=False, xlim=None):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'ctt': rf'CTT  ($g={ctt_feat+1}$)',
                'full_rank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'gray',
                'full_rank': '#984ea3'
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'full_rank': 'd'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    # compute power for each method
    powers = {}
    for method in methods:
        if method=='ctt': n_feat=ctt_feat
        else: n_feat=nys_feat
        powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

        plt.plot(vars, powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
        plt.fill_between(vars, 
                 powers[method][:,1], 
                 powers[method][:,2], 
                 alpha=0.2, color=color_dict[method])
        
    plt.hlines(0.05, xmin=-10000, xmax=110000, linestyles='dashed', colors='red', alpha=0.5, linewidth=3)
        
    plt.ylabel(r'Type-I errors', fontsize=22)
    plt.xlabel(xlabel_dict[dataset], fontsize=22)
    plt.yticks(np.arange(0, 0.125, step=0.025), fontsize=16)
    plt.xticks(fontsize=16)
    if xlim: plt.xlim(xlim)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.2), ncol=2, fontsize=24)
    plt.grid()
    plt.tight_layout()
    if file: plt.savefig(file)
    plt.show()

def plot_powervsvars_exact(results, vars, config, nys_feat=4, ctt_feat=4, legend_loc='upper left', file=False, ylim=None, xlim=None):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'ctt': rf'CTT  ($g={ctt_feat}$)',
                'fullrank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'gray',
                'fullrank': '#984ea3'
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'fullrank': 'd'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    if config['dataset']=='cg':
        idx = [0,3,5]
        vars = np.array(vars)[idx]

    # compute power for each method
    powers = {}
    for method in methods:
        
        if config['dataset']=='susy' or config['dataset']=='higgs':
            vars = vars[:3]
            
        if method=='fullrank':
            results[method] = results[method][:3,:]

            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1]])

            nvar_fullrank = results[method].shape[0]

            print(nvar_fullrank)

            plt.plot(vars[:nvar_fullrank], powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
            plt.fill_between(vars[:nvar_fullrank], 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
            
        else: 
            results[method] = results[method][idx,:,:]

            if method=='ctt': n_feat=ctt_feat
            else: n_feat=nys_feat
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

            plt.plot(vars, powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=10)
            plt.fill_between(vars, 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
    
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
        
    plt.ylabel(r'Power', fontsize=26)
    plt.xlabel(xlabel_dict[dataset], fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.3), ncol=2, fontsize=24)
    #plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.tight_layout()
    plt.grid()
    if file: plt.savefig(file)
    plt.show()



def plot_powervsvars_exactv2(results, vars, config, nys_feat=4, ctt_feat=4, legend_loc='upper left', file=False, ylim=None, xlim=None):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'ctt': rf'CTT  ($g={ctt_feat}$)',
                'fullrank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'gray',
                'fullrank': '#984ea3'
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'fullrank': 'd'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))        

    # compute power for each method
    powers = {}
    for method in methods:
        if method=='fullrank':
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1]])

            if config["dataset"]=='cg':
                idx = [0,3,5]
                fullrank_vars = np.array(vars)[idx]

                print(powers[method][:, 0])
                print([powers[method][:, 1],
                        powers[method][:, 2]])

                # errorbar replacement
                plt.errorbar(
                    fullrank_vars,
                    powers[method][:, 0],
                    yerr=[powers[method][:, 0]-powers[method][:, 1]+EPS,
                        powers[method][:, 2]-powers[method][:, 0]+EPS],
                    label=label_dict[method],
                    c=color_dict[method],
                    marker=marker_dict[method],
                    markersize=8,
                    capsize=5,
                    linestyle="",
                    alpha=0.75
                )
            else: 
                nvar_fullrank = results[method].shape[0]
                fullrank_vars = vars[:nvar_fullrank]

                plt.plot(fullrank_vars, powers[method][:,0],
                        label=label_dict[method],
                        c=color_dict[method],
                        marker=marker_dict[method],
                        markersize=10)
                plt.fill_between(fullrank_vars, 
                        powers[method][:,1], 
                        powers[method][:,2], 
                        alpha=0.2, color=color_dict[method])

            
        else: 
            if method=='ctt': n_feat=ctt_feat
            else: n_feat=nys_feat
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

            plt.plot(vars, powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=8)
            plt.fill_between(vars, 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
    
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
        
    plt.ylabel(r'Power', fontsize=26)
    plt.xlabel(xlabel_dict[dataset], fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.3), ncol=1, fontsize=24)
    #plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.tight_layout()
    plt.grid()
    if file: plt.savefig(file)
    plt.show()


def plot_powervsvars_exactv3(results, vars, config, nys_feat=4, ctt_feat=4, legend_loc='upper left', file=False, ylim=None, xlim=None):
    # plot power (at sqrt(n)) vs sample size or separation parameter

    label_dict = {
                'uniform': r'Nyström-uniform (ours, $\ell=\sqrt{n}$)',
                'rlss': r'Nyström-AKRLS (ours, $\ell=\sqrt{n}$)',
                'rff': r'RFF  ($\ell=\sqrt{n}$)',
                'ctt': rf'CTT  ($g={ctt_feat}$)',
                'fullrank': 'Exact MMD',
    }

    color_dict = {
                'uniform': '#377eb8',
                'rlss': '#ff7f00',
                'rff': '#4daf4a',
                'ctt': 'gray',
                'fullrank': '#984ea3'
    }

    marker_dict = {
                'uniform': 'v',
                'rlss': '^',
                'rff': '*',
                'ctt': '.',
                'fullrank': 'd'
    }

    xlabel_dict={
                'higgs': 'n',
                'susy': 'n',
                'cg': r'$\rho_2$',
    }

    niter = config['niter']
    dataset = config['dataset']

    methods = results.keys()

    plt.figure(figsize=(12, 10))

    if config['dataset']=='cg':
        idx = [0,3,5]
        vars = np.array(vars)[idx]
    else: 
        idx = [0,1,2]
        vars = np.array(vars)[idx]

    # compute power for each method
    powers = {}
    for method in methods:
        
        if config['dataset']=='susy' or config['dataset']=='higgs':
            vars = vars[:3]
            
        if method=='fullrank':
            results[method] = results[method][:3,:]

            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1]])

            nvar_fullrank = results[method].shape[0]

            print(nvar_fullrank)

            plt.plot(vars[:nvar_fullrank], powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=12)
            plt.fill_between(vars[:nvar_fullrank], 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
            
        else: 
            results[method] = results[method][idx,:,:]

            if method=='ctt': n_feat=ctt_feat
            else: n_feat=nys_feat
            powers[method] = np.asarray([power_interval(el, niter) for el in results[method][:,1, n_feat]])

            plt.plot(vars, powers[method][:,0], label=label_dict[method], c=color_dict[method], marker=marker_dict[method], markersize=12)
            plt.fill_between(vars, 
                    powers[method][:,1], 
                    powers[method][:,2], 
                    alpha=0.2, color=color_dict[method])
    
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
        
    plt.ylabel(r'Power', fontsize=30)
    plt.xlabel(xlabel_dict[dataset], fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.locator_params(nbins=6, axis='x')
    #plt.legend(loc=legend_loc, fontsize=18)
    plt.legend(loc=legend_loc, bbox_to_anchor=(-0.02, 1.4), ncol=1, fontsize=28)
    #plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.tight_layout()
    plt.grid()
    if file: plt.savefig(file)
    plt.show()