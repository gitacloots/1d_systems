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

def files_builder(opt1, opt2, opt3, filename, sys_description):
    f1 = filename+'.txt'
    s = ["System"+sys_description+"\n", "Option 1 = "+opt1+"\n", "Option 2 = "+opt2+"\n","Option 3 = "+opt3+"\n"]
    with open(f1, "w") as f:
        f.writelines(s)
    

def dict_builder(z_vec, pot, dens, omega, qp_vec, d, filename = "test",eta = 0.05*0.03675, sym = False):

    dict_test = {}
    return dict_test