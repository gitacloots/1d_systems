###
# This code contains methods to synthesize the dielectric properties of a specific system

import numpy as np
import diel_1D as d1
import vis_help as vh
import summarize as smr
import tools
import math
import plotly.graph_objects as go
import jellium


### Options ###
''' 1: SRF &/or MLF
    2: different potentials, densities, qp
    3: with eigenvec or without'''

def file_builder(opt1, opt2, opt3, filename, sys_description):
    f1 = filename+'.txt'
    s = ["System"+sys_description+"\n", "Option 1 = "+opt1+"\n", "Option 2 = "+opt2+"\n","Option 3 = "+opt3+"\n"]
    with open(f1, "w") as f:
        f.writelines(s)

def param_filler(index, z_vec, pot, dens, omega, qp_vec, d, options, eta = 0.05*0.03675, sym = False):
    if options[1]=='q':
        return jellium.jellium_slab(z_vec, pot, dens, qp_vec[index], omega, eta, d, sym)
    elif options[1]=='d':
        return jellium.jellium_slab(z_vec, pot, dens, qp_vec, omega, eta, d[index], sym)
    else:
        return jellium.jellium_slab(z_vec, pot[index], dens, qp_vec, omega, eta, d, sym)

def dict_builder(z_vec, pot, dens, omega, qp_vec, d, options = ('s', 'q', 'wo'), sys_description ='test' ,filename = "test",eta = 0.05*0.03675, sym = False):
    file_builder(options[0], options[1], options[2], filename, sys_description)
    dict_test = {'options':options}
    if options[1]=='q':
        dict_test['var'] = qp_vec
        var = qp_vec
        nq = len(var)
    elif options[1]=='d':
        dict_test['var'] = d
        var = d
        nq = len(var)
    elif options[1]=='p':
        dict_test['var'] = pot
        var = qp_vec
        nq = len(var[:, 0])
    
    for i in range(nq):
        dict_test[i] = {}
        sys = param_filler(i, z_vec, pot, dens, omega, qp_vec, d, options, eta, sym)
        sys.add_densities()
        if options[0]=='s':
            sys.add_surf_resp_func()
            dict_test[i]['s'] = sys.surf_resp
            if options[2] == 'wo':
                peaks, intensities = tools.find_plasmon_peaks(np.imag(sys.surf), omega)
                dict_test[i]['peak'] = peaks
                dict_test[i]['intensity'] = intensities
            else:
                peaks = tools.peak_identity(sys, opt = 'g')
                dict_test[i]['peak'] = peaks
        elif options[0]=='m':
            sys.add_loss()
            dict_test[i]['m'] = sys.loss
            if options[2] == 'wo':
                peaks = tools.find_plasmon_peaks(np.real(sys.loss), omega)
                dict_test[i]['peak'] = peaks
                dict_test[i]['intensity'] = intensities
            else:
                peaks, intensities = tools.peak_identity(sys, opt = 'l')
                dict_test[i]['peak'] = peaks
            
        else:
            sys.add_surf_resp_func()
            sys.add_loss()
            dict_test[i]['s'] = sys.surf_resp
            dict_test[i]['m'] = sys.loss
            if options[2] == 'wo':
                peaks, intensities = tools.find_plasmon_peaks(np.real(sys.loss), omega)
                dict_test[i]['peak_g'] = peaks
                dict_test[i]['intensity_g'] = intensities
                peaks, intensities = tools.find_plasmon_peaks(np.real(sys.loss), omega)
                dict_test[i]['peak_l'] = peaks
                dict_test[i]['intensity_l'] = intensities
            else:
                peaks_l, peaks_g = tools.peak_identity(sys, opt = 'lg')
                dict_test[i]['peak_l'] = peaks_l
                dict_test[i]['peak_g'] = peaks_g
    return dict_test

