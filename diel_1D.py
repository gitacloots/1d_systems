###
# This document will contain the core of the code to obtain the dielectric properties of a 1D periodic system.


import numpy as np
import tools
import math
import cmath
from numba import jit
import scipy 


### Hamiltonian & energies/wavefunctions pairs (external potential included in v_x)

def ecin(g_vec):
    g_vec = np.real(tools.inv_rev_vec(g_vec))
    return np.diag(g_vec**2/2)

def epot(v_x):
    v_g = np.fft.ifft(v_x)
    n_g = len(v_g)
    v_gmat = np.zeros((n_g, n_g), dtype = "c16")
    for i in range(n_g):
        for j in range(n_g):
            v_gmat[i, j] = v_g[j-i]   
    return v_gmat

def Hamiltonian(v_x, z_vec):
    g_v = tools.zvec_to_qvec(z_vec)
    e_cin = ecin(g_v)
    e_pot = epot(v_x)
    e_tot = e_cin+e_pot
    return (e_tot+np.conj(np.transpose(e_tot)))/2


def eig_energie(v_x, z_vec):
    ham = Hamiltonian(v_x, z_vec)
    eig_v, eig_f = np.linalg.eigh(ham)
    return eig_v, eig_f 


### dielectric properties of bulk Lindhard systems (no hamiltonian needed)
# Computation of \chi^0

ic = complex(0,1)
E0 = 1/(4*math.pi)


def im_chi0(q_vec, omega, dens=0.025):
    """Computes the imaginary part of the RPA density response function given by Lindhard"""
    npnt = len(q_vec) #size of the sampling (keep in mind that periodicity is
                  #induced in real space from the discrete sampling)
    n_w = len(omega) #Number of frequency sampled
    chi0_q = np.zeros((n_w, npnt), dtype = "c16")
    e_f = (1/2)*(3*math.pi**2*dens)**(2/3)
    for i in range(n_w):
        w_i = omega[i]
        for j in range(npnt):
            q_norm = abs(q_vec[j])
            if q_norm==0:
                continue
            e_minus = (w_i-(q_norm**2)/2)**2*(2/(q_norm**2))*1/4
            if e_minus<=(e_f-w_i):
                chi0_q[i, j]=1/(2*math.pi)*(1/q_norm)*w_i
            elif e_f >= e_minus >= e_f-w_i:
                chi0_q[i, j]=1/(2*math.pi)*(1/q_norm)*(e_f-e_minus)
            else:
                continue
    return -chi0_q


def re_chi0(q_vec, omega, dens=0.025):
    """Computes the imaginary part of the RPA density response function given by Lindhard"""
    npnt = len(q_vec)
    n_w = len(omega)
    chi0_q = np.zeros((n_w, npnt), dtype = "c16")
    k_f = (3*math.pi**2*dens)**(1/3)
    pref = -4/(2*math.pi)**2*k_f
    for i in range(n_w):
        w_i = omega[i]
        k_w = (2*w_i)**(1/2)
        for j in range(npnt):
            q_norm = abs(q_vec[j])
            if q_norm==0:
                continue
            t_1 = 1-(1/4)*((k_w**2-q_norm**2)**2/(k_f**2*q_norm**2))
            t_21 = (k_w**2-2*k_f*q_norm-q_norm**2)/(k_w**2+2*k_f*q_norm-q_norm**2)
            t_22 = math.log(abs(t_21))
            t_3 = 1-(1/4)*((k_w**2+q_norm**2)**2/(k_f**2*q_norm**2))
            t_41 = (k_w**2+2*k_f*q_norm+q_norm**2)/(k_w**2-2*k_f*q_norm+q_norm**2)
            t_42 = math.log(abs(t_41))
            chi0_q[i, j] = 1/2+(k_f/(4*q_norm))*(t_1*t_22+t_3*t_42)
    return pref*chi0_q

def chi0q(q_vec, omega, dens = 0.025):
    """Computes the RPA density response function in reciprocal space
     following the Lindhard formula"""
    rchi = re_chi0(q_vec, omega, dens)
    ichi = im_chi0(q_vec, omega, dens)
    chi0_q_in = rchi+ic*ichi
    chi0_q_out = chi0_mat(chi0_q_in)
    return chi0_q_out

@jit(nopython = True, parallel=True)
def chi0_mat(chi0_q):
    """Transform chi0_q (shape: n_w, nq) in a chi0qq
     (serie of diagonal matrices; shape: n_w, nq, nq)"""
    n_w, n_q = chi0_q.shape
    chi0_qq = np.zeros((n_w, n_q, n_q), dtype = "c16")
    for i in range(n_w):
        chi0_qq[i] = np.diag(chi0_q[i])
    return chi0_qq   



@jit(nopython = True, parallel=True)
def chiq_mat(chi0_q, q_vec):
    """Computes the density response function with coulomb effect with a matrix as input."""
    n_w, n_q = chi0_q.shape[0], chi0_q.shape[1]
    chiq_m = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    for i in range(n_q):
        if q_vec[i] == 0:
            continue
        coulomb[i, i] = 4*math.pi/(q_vec[i]**2)
    #print(np.diag(coulomb))
    print("coulomb ok")
    chi_to_inv = np.zeros((n_q, n_q), dtype = "c16")
    for i in range(n_w):
        for j in range(n_q):
            if chi0_q[i, j, j] == 0:
                chiq_m[i, j, j] = chi0_q[i, j, j]
            else:
                chi_to_inv[j, j] = (chi0_q[i, j, j])**(-1)-coulomb[j, j]
                chiq_m[i, j, j] = chi_to_inv[j, j]**(-1)
    return chiq_m

### Dielectric properties of 3D Systems represented by a 1D potential
# Computation of \chi^0(q_||, \omega; z, z')

def pre_run_chi0(v_pot, z_vec, dens, d_sys):
    n_z = len(v_pot)
    energies, bands = eig_energie(v_pot, z_vec)
    index = list(np.argsort(energies))
    bands_sorted = bands[:, index]
    energies = (energies[index])
    #energies = np.append(np.array([0]),energies)
    e_f, nmax = ef_2D_full(dens, d_sys, energies)
    bands_z = np.zeros((bands.shape), dtype = "c16")
    for i in range(n_z):
        bands_z[:, i] = np.fft.ifft((bands_sorted[:, i]))
        N = tools.func_norm(z_vec, bands_z[:, i])
        bands_z[:, i] = bands_z[:, i]/N
        #bands_z[:, i] = bands_z[:, i]/np.linalg.norm(bands_z[:, i])
    return energies, bands_z, e_f, nmax

@jit(nopython = True)
def ef_2D_full(n, d, e_vec):
    "Find the fermi level in a slab"
    n = n*d
    if n == 0:
        raise ValueError("the fermi level is not uniquely defined if the density is zero")
    e_max = e_vec[0]+0.1
    e_min = e_vec[0]
    e_tot = e_vec[0]
    i = 1
    while np.real(e_max) > np.real(e_min):
        e_max = (math.pi*n + e_tot)/i
        if i > len(e_vec):
            raise ValueError("Number of states too low, you should add more states in order to find the Fermi level")
        e_min = e_vec[i]
        e_tot += e_min
        i += 1
    return e_max, i-1

@jit(debug = True, nopython = True, parallel=True)
def chi0wzz_slab_jellium_with_pot(q_p, energies, bands_z, omega, e_f, nmax, nband_tot, eta, sym = True):
    """Computes the density response function as found by Eguiluz with the slab represented as an infinite well"""
    #print("Computation from potential started")
    n_w = len(omega)
    #energies = energies[1::]
    #nmax = nmax-1
    n_z = len(energies)
    
    wff = np.zeros((n_z, n_z, nband_tot), dtype = "c16")
    #wff2 = np.zeros((n_z, n_z, n_z), dtype = "c16")
    for i in range(n_z):
        for j in range(n_z):
            for k in range(nband_tot):
                wff[i, j, k] = bands_z[i, k]*np.conj(bands_z[j,k])
                #wff1[i, j, k] = np.conj(bands_z[i, k])*bands_z[j,k]
                #wff2[i, j, k] = bands_z[i, k]*np.conj(bands_z[j,k])
    
    if sym:
        n_half = math.ceil(n_z/2)
    else:
        n_half = n_z
    chi0wzz = np.zeros((n_w, n_half, n_z), dtype = "c16")
    for i in range(n_w):
        fll = np.zeros((nmax, nband_tot), dtype = "c16")
        for j in range(nmax):
            for k in range(4*nmax):
                fll[j, k] = f_ll_pot(q_p, omega[i], energies[j], energies[k], e_f, eta)
        for j in range(n_half):
            for k in range(n_z):
                for l in range(nmax):
                    wffi = np.conj(wff[j, k, l])
                    #wffi2 = wff2[j, k, l]
                    for m in range(nband_tot):
                        wffj = wff[j, k, m]
                        #wffj2 = np.conj(wff2[j, k, m])
                        chi0wzz[i, j, k]+=(wffi*wffj)*fll[l, m]
                        #chi0wzz[i, j, k]+=(wffi1*wffj1+wffi2*wffj2)*fll[l, m]/2 
    #chi0wzz = chi0wzz*n_z**2/d_sys**2                    
    return chi0wzz


@jit(debug = True, nopython = True)
def a_ll_pot(q_p, e_l1, e_l2):
    """Segment of the prefactor of Eguiluz1985"""
    return (q_p**2)/2-(e_l1-e_l2)


@jit(debug = True, nopython = True)
def f_ll_pot(q_p, omega, e_l1, e_l2, e_f, eta):
    """Compute the prefactor from the formula of Eguiluz1985"""
    a_l1l2 = a_ll_pot(q_p, e_l1, e_l2)
    if q_p == 0:
        pre_factor = (e_f-e_l1)/(math.pi)
        return -pre_factor*(1/(a_l1l2+omega+ic*eta)+1/(a_l1l2-omega-ic*eta))
    else:
        k_l = cmath.sqrt(2*(e_f-e_l1))
        return -1/(math.pi*q_p**2)*(2*a_l1l2+ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2-omega-ic*eta)**2)-ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2+omega+ic*eta)**2))


### Computation of the electronic density of the system

def density(energies, bands, e_f):
    index = 0
    npnt = len(energies)
    dens = np.zeros((npnt))
    while energies[index] < e_f:
        bands_dot = bands[:, index]*np.conj(bands[:, index])
        if np.max(np.imag(bands_dot))>1e-18:
            print(np.max(np.imag(bands_dot)))
            raise ValueError("It seems the eigen function are not orthonormal")
        dens += (e_f-energies[index])*np.real(bands_dot)
        index += 1
    return 1/math.pi*dens

### Other dielectric properties of 1D potential systems

@jit(debug = True, nopython = True, parallel=True)
def coulomb2d(q_paral, z_pot):
    nz = len(z_pot)
    delta = np.zeros((nz, nz))
    z_pot = tools.center_z(z_pot)
    for i in range(nz):
        for j in range(nz):
            delta[i, j] = np.abs(z_pot[i]-z_pot[j])
    return 2*math.pi/q_paral*np.exp(-q_paral*delta)

@jit(debug = True, nopython = True, parallel=True)
def eps_inv(eps_2d):
    nw, nz = eps_2d.shape[0], eps_2d.shape[1]
    eps_2d_i = np.zeros((nw, nz, nz), dtype = "c16")
    for i in range(nw):
        eps_2d_i[i] = np.linalg.inv(eps_2d[i])
    return eps_2d_i

#@jit(debug = True, nopython = True, parallel=True)
def dielectric_2d(chi0, coulomb_2d, dz):
    nw, nz = chi0.shape[0], chi0.shape[1]
    diel2d = np.zeros((nw, nz, nz), dtype = "c16")
    for i in range(nw):
        diel2d[i] = np.diag(np.ones(nz))/dz-np.matmul(coulomb_2d, chi0[i])*dz
    return diel2d

#@jit(debug = True, nopython = True, parallel=True)
def dielectric_inv_2d(chi_2d, coulomb_2d, dz):
    nw, nz = chi_2d.shape[0], chi_2d.shape[1]
    diel_i2d = np.zeros((nw, nz, nz), dtype = "c16")
    for i in range(nw):
        diel_i2d[i] = np.diag(np.ones(nz))/dz+np.matmul(coulomb_2d, chi_2d[i])*dz
    return diel_i2d

@jit(debug = True, nopython = True, parallel=True)
def chi_from_epsilon(eps_2d, coulomb_2d, dz):
    nw, nz = eps_2d.shape[0], eps_2d.shape[1]
    eps_inv_2d = eps_inv(eps_2d)
    chi_2d = np.zeros((nw, nz, nz), dtype = "c16")
    coulomb_inv = np.linalg.inv(coulomb_2d)
    for i in range(nw):
        chi_2d[i] = np.matmul(coulomb_inv, eps_inv_2d[i]-np.diag(np.ones(nz))/dz)*dz
    return chi_2d

##@jit(debug = True, nopython = True, parallel=True)
def chi_from_epsilon_inv(eps_inv_2d, coulomb_2d, dz):
    nw, nz = eps_inv_2d.shape[0], eps_inv_2d.shape[1]
    chi_2d = np.zeros((nw, nz, nz), dtype = "c16")
    coulomb_inv = np.linalg.inv(coulomb_2d)
    for i in range(nw):
        chi_2d[i] = np.matmul(coulomb_inv, eps_inv_2d[i]-np.diag(np.ones(nz))/dz)*dz
    return chi_2d

#@jit(debug = True, nopython = True, parallel=True)
def screened_int_from_chi(chi_2d, coulomb_2d, dz):
    nw, nz = chi_2d.shape[0], chi_2d.shape[1]
    w_2d = np.zeros((nw, nz, nz), dtype = "c16")
    for i in range(nw):
        w_2d[i] = coulomb_2d + np.matmul(np.matmul(coulomb_2d, chi_2d[i]), coulomb_2d)*dz**2
    return w_2d

### In reciprocal space:
def epsilon(chi0qgg, q_vec, q_p):
    """Computes the dielectric function of a slab associated with chi0qgg"""
    n_w, n_q = chi0qgg.shape[0], chi0qgg.shape[1]
    eps_out = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    if q_p == 0:
        for i in range(1, n_q):
            if q_vec[i]==0:
                continue
            coulomb[i, i] = 4*math.pi*np.power((q_vec[i]), -2)
    else:
        for i in range(n_q):
            coulomb[i, i] = 4*math.pi/(q_vec[i]**2+q_p**2)
    for i in range(n_w):
        eps_out[i] = np.diag(np.ones(n_q))-np.matmul(coulomb, chi0qgg[i])
    return eps_out

### If central symmetry:
@jit(nopython = True, parallel=True)
def sym_chi_slab(chi0wzz):
    """Symmetrizes chi0wzz of the first spatial components only spans in 
    the first half of the sampled area"""
    n_w, n_z1, n_z2 = chi0wzz.shape
    if n_z1 == n_z2:
        return chi0wzz
    else:
        chi0wzz_slab = np.zeros((n_w, n_z2, n_z2), dtype = "c16")
        for i in range(n_w):
            chi0wzz_slab[i, 0:n_z1, 0:n_z2] = chi0wzz[i, :, :]
            for j in range(n_z1):
                chi0wzz_slab[i, n_z1-1+j, :] = chi0wzz[i, n_z1-1-j, :][::-1]
        return chi0wzz_slab
    
###Surface Response function

def surface_response_function(z, drf, qp, delta):
    nz = len(z)
    nw = drf.shape[0]
    phase_fac = np.zeros((nz, nz))
    for i in range(nz):
        for j in range(nz):
            phase_fac[i, j] = z[i]+z[j]
    phase_fac = np.exp(qp*phase_fac)
    g = np.zeros((nw), dtype = "c16")
    for i in range(nw):
        g[i] = np.sum(np.multiply(phase_fac, drf[i]))
    return -2*math.pi/qp*g*delta**2

### Macroscopic loss function

@jit(nopython = True, parallel=True)
def weight_full(eps_wgg):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec_dual[0] = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    vec_dual_p = np.linalg.inv(vec_p)
    """for i in range(n_q):
        vec_dual_p[i, :] = vec_dual_p[i, :]/np.linalg.norm(vec_dual_p[i, :])"""
    #vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
    vec_dual[0] = vec_dual_p
    vec[0] = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0, 0,:],(np.transpose(vec_dual[0, :,0])))
    for i in range(1, n_w):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        vec_dual_p = np.linalg.inv(vec_p)
        """for j in range(n_q):
            vec_dual_p[j, :] = vec_dual_p[j, :]/np.linalg.norm(vec_dual_p[j, :])"""
        ####
        vec[i] = vec_p
        vec_dual[i] = vec_dual_p
        eig[i] = eig_all[i]
        weights_p[i]=np.multiply(vec[i, 0,:],(np.transpose(vec_dual[i, :,0])))
    return weights_p, eig, vec, vec_dual



@jit(nopython = True, parallel=True)
def loss_func_majerus(weights_p, eig):
    """Computes the loss function as given in the thesis of Bruno Majerus"""
    n_w, n_q = eig.shape
    loss_func = np.zeros((n_w), dtype = "c16")
    for i in range(n_q):
        loss_func_i = -np.imag(np.power(eig[:, i], -1))
        weight_i = weights_p[:, i]
        loss_func += np.multiply(loss_func_i, weight_i)
    return loss_func

def loss_full_slab_wov(diel, z_2, q_p):
    """Gives the spectra from the density response function"""
    diel_wov = sym_chi_slab(diel)
    q_vec = tools.zvec_to_qvec(z_2)
    diel_wov_q = tools.fourier_inv(diel_wov, z_2)
    eps = epsilon(diel_wov_q, np.real(tools.inv_rev_vec(q_vec)), q_p)
    weights, eig_q, vec, vec_dual = weight_full(eps)
    loss = loss_func_majerus(weights, eig_q)
    return loss, weights, eig_q, eps, vec, vec_dual