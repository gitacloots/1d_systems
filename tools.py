###
# This code contains transversal subroutines to manage the vector position (units and origin)
# as well as the Fourier Transform

import numpy as np
import math
import cmath
from numba import jit

### Vector conversions

@jit(nopython = True)
def qvec_to_zvec(q_vec):
    """Enter the q vector (-q, q) with the point q=0 so always an odd number of point"""
    npnt = len(q_vec)
    center = math.floor(npnt/2)+1
    qmin = q_vec[center]
    return np.linspace(0, 2*math.pi/qmin, npnt)


@jit(nopython = True)
def zvec_to_qvec(z_vec):
    """enter the z vector (0, zmax)"""
    npnt = len(z_vec)
    z_sort = np.sort(np.abs(z_vec))
    if z_sort[0] != 0:
        raise ValueError("Watch out, the given vector does not contain z = 0 which is essential in this implementation to obtain the coreect vector in reciprocal space")
    zmin = z_sort[1]
    qnosym = np.linspace(-math.pi/zmin, math.pi/zmin, npnt)
    qsym = np.zeros((npnt))
    for i in range(math.floor(npnt/2)+1):
        q_vec = (-qnosym[i]+qnosym[npnt-i-1])/2
        qsym[i] = -q_vec
        qsym[npnt-i-1] = q_vec
    return qsym


@jit(nopython = True)
def inv_rev_vec(y_in):
    """In : [-3, -2, -1, 0, 0, 1, 2, 3] // [-3, -2, -1, 0, 1, 2, 3]
       Out : [0, 1, 2, 3, -3, -2, -1, -0] // [0, 1, 2, 3, -3, -2, -1]"""
    npnt = y_in.size
    y_out = np.zeros((npnt), dtype = "c16")
    if npnt%2==0:
        mid = math.floor(npnt/2)
        y_out[0:mid] = y_in[mid:npnt]
        y_out[mid:npnt] = y_in[0:mid]
    else:
        mid = math.floor(npnt/2)+1
        y_out[0:mid] = y_in[mid-1:npnt]
        y_out[mid:npnt] = y_in[0:mid-1]
    return y_out

#@jit(nopython = True)
def center_z(z_vec):
    """Set the origin to the center of the slab rather than on the far left of the computed area"""
    npnt = len(z_vec)
    z_half = max(z_vec)/2
    z = np.linspace(-z_half, z_half, npnt)
    return (z-z[::-1])/2

### Normalization
@jit(nopython = True, parallel=True)
def func_norm(x, fx):
    return cmath.sqrt(np.sum(np.multiply(np.conj(fx), fx))*np.abs(x[0]-x[1]))


### Extract diagonal content

@jit(nopython = True)
def extract_diag(matwdd):
    """"Extracts the diagonal of 3D matrix by extracting its diagonals"""
    n_w, n_d = matwdd.shape[0], matwdd.shape[1]
    mat_out = np.zeros((n_w, n_d), dtype = "c16")
    for i in range(n_w):
        mat_out[i] = np.diag(matwdd[i])
    return mat_out


### Fourier transforms
def fourier_inv(chi0wzz, z_vec):
    """Computes chi0qgg from chi0wzz"""
    n_w, nz1, nz2 = chi0wzz.shape
    if nz1 !=nz2:
        raise ValueError("The matrix must have the same dimension nz1 and nz2")
    chi0wzq2 = np.zeros((n_w, nz1, nz2), dtype = "c16")
    for i in range(n_w):
        for j in range(nz1):
            chi0wzq2[i, j, :] = np.fft.fft(chi0wzz[i, j, :], norm = "ortho")
    chi0wq1q2 = np.zeros((n_w, nz1, nz2), dtype = "c16")
    for i in range(n_w):
        for j in range(nz2):
            chi0wq1q2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j], norm = "ortho")
    return chi0wq1q2/nz2*max(z_vec)

def fourier_dir(matwgg):
    """Performs the Fourier Transform to go from chi0wqq to chi0wzz"""
    n_w, nq1, nq2 = matwgg.shape
    matwzq2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq1):
            matwzq2[i, :, j] = np.fft.ifft(matwgg[i, :, j])
    matwz1z2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq2):
            matwz1z2[i, j, :] = np.fft.fft(matwzq2[i, j, :])
    matwz1z2_out = matwz1z2
    '''matwz1z2_out = np.zeros((n_w, nq1, nq2), dtype = "c16")    
    for i in range(n_w):
        matwz1z2_out[i] = (matwz1z2[i, :, :]+np.transpose(matwz1z2[i, :, :]))/2'''
    return matwz1z2_out

### Information from the dielectric properties (peak position, spatial description of plasmon mode)
def peak_identity(system, opt = 'lg'):
    nw = len(system.omega)
    weights = np.real(system.weights)
    eig_q_inv = -np.imag(np.power(system.eig_q, -1))
    weighted_eig_q = np.multiply(weights, eig_q_inv)
    if opt == 'l':
        l_w = np.real(system.loss)
        dl_w = np.gradient(l_w, system.omega)
        max_intl = np.max(l_w)
        peak_l = {}
        intensity_l = {}
        for i in range(1, nw):
            if dl_w[i-1]>0 and dl_w[i]<=0:
                #print("in")
                peak_l[i] = system.omega[i]
                intensity_l[i] = l_w[i]/max_intl
        peaks_l = {}
        for i in peak_l.keys():
            index_m1 = np.argmax(weighted_eig_q[i])
            index_m2 = np.argmax(system.weights[i])
            peaks_l[i] = {"freq" : np.round(system.omega[i]*27.211, 3), "vec": system.vec[i, :, index_m1], "vec_dual": system.vec_dual[i, index_m1, :], "intensity" : intensity_l[i], "mode_maj" : index_m1, "weight_mode_maj" : weights[i, index_m1], "mode2" : np.array([index_m2, weights[i, index_m2]])}
        if opt == 'l':
            return peaks_l
    elif opt == 'g':
        g_w = np.imag(system.surf_resp)
        dg_w = np.gradient(g_w, system.omega)
        max_intg = np.max(g_w)
        peak_g = {}
        intensity_g = {}
        for i in range(1, nw):
            if dg_w[i-1]>0 and dg_w[i]<=0:
                #print("in")
                peak_g[i] = system.omega[i]
                intensity_g[i] = g_w[i]/max_intg
        
        peaks_g = {}
        for i in peak_g.keys():
            index_m1 = np.argmax(eig_q_inv[i])
            index_m2 = np.argmax(system.weights[i])
            peaks_g[i] = {"freq" : np.round(system.omega[i]*27.211, 3), "vec": system.vec[i, :, index_m1], "vec_dual": system.vec_dual[i, index_m1, :], "intensity" : intensity_g[i], "mode_maj" : index_m1, "weight_mode_maj" : weights[i, index_m1], "mode2" : np.array([index_m2, weights[i, index_m2]])}
        if opt == 'g':
            return peaks_g
    else:
        peaks_l = peak_identity(system, 'l')
        peaks_g = peak_identity(system, 'g')
    return peaks_l, peaks_g

def find_plasmon_peaks(g_w, omega):
    dg_w = np.gradient(g_w, omega)
    #print(dg_w)
    nw = len(omega)
    max_int = np.max(g_w)
    #print(nw)
    peak = {}
    intensity = {}
    for i in range(1, nw):
        if dg_w[i-1]>0 and dg_w[i]<=0:
            #print("in")
            peak[i] = omega[i]
            intensity[i] = g_w[i]/max_int
    return peak, intensity

