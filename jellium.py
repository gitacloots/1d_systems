###
# This code builds a jellium object that can use the subroutines of all the other documents to
# provide the dielectric information of 1D periodic systems
# 

import numpy as np
import diel_1D as d1
import vis_help as vh
import summarize as smr
import tools
import math
import plotly.graph_objects as go


class jellium_slab:
    #This object only works for symmetrical slabs!!
    def __init__(self, z_vec, pot, dens, qp = 0, omega = np.array([0]), eta = 0.05*0.03675, d = [], dens_bg = [], pres_model = False, sym = False):
        self.z = z_vec
        self.dens_3d = dens
        self.tot_pot = pot
        self.nz = len(self.z)
        self.delta_z = np.abs(self.z[0]-self.z[1])
        if d == []:
            self.width = max(self.z)
            self.nz_slab = self.nz
        else:
            self.width, self.nz_slab = d, round(d/self.delta_z)
        self.dens_2d = self.width*self.dens_3d
       
        self.qp = qp
        self.omega = omega
        self.damping = eta
        self.nmax = 0
        self.plasma_freq = math.sqrt(self.dens_3d*4*math.pi) 
        self.dens_bg = dens_bg
        self.sym = sym

        self.drf0 = np.array([])
        self.coulomb = np.array([])
        self.df = np.array([])
        self.df_inv = np.array([])
        self.drf = np.array([])
        self.scr_int = np.array([])
        self.loss = np.array([])
        self.surf_resp = np.array([])
        if pres_model:
            print("The input data for this model are:\n length of the system : "+str(np.round(max(self.z)-min(self.z), 3))+' Bohr (one point each '+str(np.round(self.delta_z, 3))+' Bohr)\n range of frequency:'+str(np.round(min(self.omega), 3))+'-'+str(np.round(max(self.omega), 3))+' Ha with '+str(len(self.omega))+' points\n jellium density : '+str(self.dens_3d)+' Bohr^-3\n total width of the jellium : '+str(np.round(self.width, 3))+' Bohr\n qp : '+str(np.round(self.qp, 3))+' Bohr^-1\n eta : '+str(np.round(self.damping, 3))+' Ha')
            self.add_densities()
            self.show_densities()
            self.show_pot()
    def add_shrodinger_sol(self):
        self.energies, self.bands, self.e_f, self.nmax = d1.pre_run_chi0(self.tot_pot, self.z, self.dens_3d, self.width)

    def add_densities(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.dens_elec = d1.density(self.energies, self.bands, self.e_f)
        if len(self.dens_bg) == 0:
            z = tools.center_z(self.z)
            dens_b = np.zeros(self.nz)
            for i in range(self.nz):
                if np.abs(z[i])<self.width/2:
                    dens_b[i] = 1
            self.dens_bg = dens_b*(np.sum(self.dens_elec)*self.delta_z)/(np.sum(dens_b)*self.delta_z)

    '''def add_energies(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.kin_energy = kinetic_energy(self.bands, self.nmax, self.z)
        self.hartree_energy = hartree_energy(self.dens_elec, self.dens_bg, self.z)
        self.xc_energy = xc_Energy(self.dens_elec, self.z)
        self.tot_energy = self.kin_energy+self.hartree_energy+self.xc_energy
    '''
    def update_omega(self, omega):
        self.omega = omega
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()
        

    def update_qp(self, qp):
        self.qp = qp
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()

    def update_damping(self, damping):
        self.damping = damping
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()


    def add_drf0(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.drf0 = d1.chi0wzz_slab_jellium_with_pot(self.qp, self.energies, self.bands, self.omega, self.e_f, self.nmax, 4*self.nmax, self.damping, sym = self.sym)
        if self.sym:
            self.drf0 = d1.sym_chi_slab(self.drf0)

    def add_diel_properties(self):
        self.coulomb = d1.coulomb2d(self.qp, self.z)
        if self.drf0.size == 0:
            self.add_drf0()
        self.df = d1.dielectric_2d(self.drf0, self.coulomb, self.delta_z)
        self.df_inv = d1.eps_inv(self.df)
        self.drf = d1.chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
        self.scr_int = d1.screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def add_coulomb(self):
        self.coulomb = d1.coulomb2d(self.qp, self.z)
    
    def add_df(self):
        if self.drf0.size == 0:
            self.add_drf0()
        if self.coulomb.size == 0:
            self.add_coulomb()
        self.df = d1.dielectric_2d(self.drf0, self.coulomb, self.delta_z)

    def add_df_inv(self):
        if self.df.size == 0:
            self.add_df()
        self.df_inv = d1.eps_inv(self.df)
    
    def add_drf(self):
        if self.df_inv.size == 0:
            self.add_df_inv()
        self.drf = d1.chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
    
    def add_scr_int(self):
        if self.drf.size == 0:
            self.add_drf()
        self.scr_int = d1.screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def update_diel_properties(self):
        self.coulomb = d1.coulomb2d(self.qp, self.z)
        self.add_drf0()
        self.df = d1.dielectric_2d(self.drf0, self.coulomb, self.delta_z)
        self.df_inv = d1.eps_inv(self.df)
        self.drf = d1.chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
        self.scr_int = d1.screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def update_coulomb(self):
        self.coulomb = d1.coulomb2d(self.qp, self.z)
    
    def update_df(self):
        self.add_drf0()
        self.coulomb = self.update_coulomb()
        self.df = d1.dielectric_2d(self.drf0, self.coulomb, self.delta_z)

    def update_df_inv(self):
        self.df = self.update_df()
        self.df_inv = d1.eps_inv(self.df)
    
    def update_drf(self):
        self.df_inv = self.update_df_inv()
        self.drf = d1.chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
    
    def update_scr_int(self):
        self.drf = self.update_drf()
        self.scr_int = d1.screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def clear_diel(self, diel):
        if diel == "drf0":
            self.drf0 = []
        elif diel == "drf":
            self.drf = []
        elif diel == "df":
            self.df = []
        elif diel == "df_inv":
            self.df_inv = []
        elif diel == "scr_int":
            self.scr_int = []
        else:
            print("no dielectric properties corresponding to ", diel)

    def add_loss(self):
        if self.drf0.size == 0:
            self.add_drf0()
        self.loss, self.weights, self.eig_q, self.eps, self.vec, self.vec_dual = d1.loss_full_slab_wov(self.drf0, self.z, self.qp) 


    def add_surf_resp_func(self):
        if self.drf.size == 0:
            self.add_drf()
        self.surf_resp = d1.surface_response_function(self.z, self.drf, self.qp, self.delta_z)

    def show_pot(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.tot_pot), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Potential of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$V_{tot}$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_density_elec(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_elec), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Electronic density of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_density_bg(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_bg), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Background density of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho^+(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_densities(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"

        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_elec), name = r"$\rho^-$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_bg), name = r"$\rho^+$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Densities of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_1d_drf0(self, w, i):
        vh.show_1d(self, self.drf0, self.z, self.width, w, i, r'$\chi^0\text{ for a perturbation in }z_1$', r'$\Re{\chi^0}$', r'$\Im{\chi^0}$', r'$\chi^0(\omega, z_1, z)$')
        
    def show_1d_drf(self, w, i):
        vh.show_1d(self, self.drf, self.z, self.width, w, i, r'$\chi\text{ for a perturbation in }z_1$', r'$\Re{\chi}$', r'$\Im{\chi}$', r'$\chi(\omega, z_1, z)$')
    
    def show_1d_df(self, w, i):
        vh.show_1d(self, self.df, self.z, self.width, w, i, r'$\epsilon\text{ for a perturbation in }z_1$', r'$\Re{\epsilon}$', r'$\Im{\epsilon}$', r'$\epsilon(\omega, z_1, z)$')

    def show_1d_df_inv(self, w, i):
        vh.show_1d(self, self.df_inv, self.z, self.width, w, i, r'$\epsilon^{-1}\text{ for a perturbation in }z_1$', r'$\Re{\epsilon^{-1}}$', r'$\Im{\epsilon^{-1}}$', r'$\epsilon^{-1}(\omega, z_1, z)$')

    def show_1d_scr_int(self, w, i):
        vh.show_1d(self, self.df_inv, self.z, self.width, w, i, r'$W\text{ for a perturbation in }z_1$', r'$\Re{W}$', r'$\Im{W}$', r'$W(\omega, z_1, z)$')


    def show_1d_drf0_omega(self, i):
        vh.show_1d_omega(self.drf0, self.omega, self.z, i, r'$\chi^0 \text{ for a perturbation and response in z}$', r'$\Re{\chi^0}$', r'$\Im{\chi^0}$', r'$\chi^0(\omega, z, z)$')


    def show_1d_drf_omega(self, i):
        vh.show_1d_omega(self.drf, self.omega, self.z, i, r'$\chi \text{ for a perturbation and response in z}$', r'$\Re{\chi}$', r'$\Im{\chi}$', r'$\chi(\omega, z, z)$')

    
    def show_1d_df_omega(self, i):
        vh.show_1d_omega(self.df, self.omega, self.z, i, r'$\epsilon \text{ for a perturbation and response in z}$', r'$\Re{\epsilon}$', r'$\Im{\epsilon}$', r'$\epsilon(\omega, z, z)$')

    def show_1d_df_inv_omega(self, i):
        vh.show_1d_omega(self.df_inv, self.omega, self.z, i, r'$\epsilon^{-1} \text{ for a perturbation and response in z}$', r'$\Re{\epsilon^{-1}}$', r'$\Im{\epsilon^{-1}}$', r'$\epsilon^{-1}(\omega, z, z)$')

    def show_1d_scr_int_omega(self, i):
        vh.show_1d_omega(self.scr_int, self.omega, self.z, i, r'$W \text{ for a perturbation and response in z}$', r'$\Re{W}$', r'$\Im{W}$', r'$W(\omega, z, z)$')


    def show_map_drf0(self, w, compl = False):
        if compl:
            vh.show_map(np.imag(self.drf0), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\chi^0}(z_1, z_2)$')
        else:
            vh.show_map(np.real(self.drf0), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\chi^0}(z_1, z_2)$')

    def show_map_drf0_omega(self, compl = False):
        if compl:
            vh.show_map_omega(np.imag(self.drf0), self.z, self.omega, self.width, self.nz, r'$\Im{\chi^0}(\omega, z, z)$')
        else:
            vh.show_map_omega(np.real(self.drf0), self.z, self.omega, self.width, self.nz, r'$\Re{\chi^0}(\omega, z, z)$')


    def show_map_drf(self, w, compl = False):
        if compl:
            vh.show_map(np.imag(self.drf), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\chi}(z_1, z_2)$')
        else:
            vh.show_map(np.real(self.drf), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\chi}(z_1, z_2)$')

    def show_map_drf_omega(self, compl = False):
        if compl:
            vh.show_map_omega(np.imag(self.drf), self.z, self.omega, self.width, self.nz, r'$\Im{\chi}(\omega, z, z)$')
        else:
            vh.show_map_omega(np.real(self.drf), self.z, self.omega, self.width, self.nz, r'$\Re{\chi}(\omega, z, z)$')

    def show_map_df(self, w, compl = False):
        if compl:
            vh.show_map(np.imag(self.df), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\epsilon}(z_1, z_2)$')
        else:
            vh.show_map(np.real(self.df), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\epsilon}(z_1, z_2)$')

    def show_map_df_omega(self, compl = False):
        if compl:
            vh.show_map_omega(np.imag(self.df), self.z, self.omega, self.width, self.nz, r'$\Im{\epsilon}(\omega, z, z)$')
        else:
            vh.show_map_omega(np.real(self.df), self.z, self.omega, self.width, self.nz, r'$\Re{\epsilon}(\omega, z, z)$')

    def show_map_df_inv(self, w, compl = False):
        if compl:
            vh.show_map(-np.imag(self.df_inv), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$-\Im{\epsilon^{-1}}(z_1, z_2)$')
        else:
            vh.show_map(np.real(self.df_inv), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\epsilon^{-1}}(z_1, z_2)$')

    def show_map_df_inv_omega(self, compl = False):
        if compl:
            vh.show_map_omega(-np.imag(self.df_inv), self.z, self.omega, self.width, self.nz, r'$-\Im{\epsilon^{-1}}(\omega, z, z)$')
        else:
            vh.show_map_omega(np.real(self.df_inv), self.z, self.omega, self.width, self.nz, r'$\Re{\epsilon^{-1}}(\omega, z, z)$')

    
    def show_map_scr_int(self, w, compl = False):
        if compl:
            vh.show_map(-np.imag(self.scr_int), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$-\Im{W}(z_1, z_2)$')
        else:
            vh.show_map(np.real(self.scr_int), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{W}(z_1, z_2)$')

    def show_map_scr_int_omega(self, compl = False):
        if compl:
            vh.show_map_omega(-np.imag(self.scr_int), self.z, self.omega, self.width, self.nz, r'$-\Im{W}(\omega, z, z)$')
        else:
            vh.show_map_omega(np.real(self.scr_int), self.z, self.omega, self.width, self.nz, r'$\Re{W}(\omega, z, z)$')

    def show_loss(self):
        fig = go.Figure()
        xtitle = r"$\omega \text{ [eV]}$"
        fig.add_trace(go.Scatter(x = self.omega*27.211, y = np.real(self.loss), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Loss function', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$L(\omega)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_surf_resp_func(self):
        fig = go.Figure()
        xtitle = r"$\omega \text{ [eV]}$"
        fig.add_trace(go.Scatter(x = self.omega*27.211, y = np.imag(self.surf_resp), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Surface response function', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$L(\omega)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_eig_modes(self, n_mode):
        if type(n_mode) == int:
            n_mode = range(n_mode)
        
        fig = go.Figure()
        xtitle = r"$\omega \text{ [eV]}$"
        for i in n_mode:
            fig.add_trace(go.Scatter(x = self.omega*27.211, y = -np.imag(np.power(self.eig_q[:, i], -1)), name = r"mode "+str(i), mode = "lines+markers", marker=dict(
                size=5,
                ),))
        fig.update_layout(title_text = r'Loss function', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$L(\omega)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()
    
    def show_spat_mode(self, w, i):
        fig = go.Figure()
        xtitle = r"$z \text{[Bohr]}$"
        ytitle = r"$V_i(z), \rho_i(z)$"
        title = r"$\text{Eigenfunctions associated with the visible mode in the EELS spectra}$"
        y1 = np.real(np.fft.fft(self.vec[w, :, i]))
        y2 = np.real(np.fft.ifft(self.vec_dual[w, i, :], norm  = "forward"))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y1 , showlegend = False, name = r"$V_i(z)$", mode = "lines",marker=dict(
            size=5,color = "blue",
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y =  y2, showlegend = False, name = r"$\rho_i(z)$", mode = "lines",marker=dict(
            size=5,color = "red",
            ),))
        fig.add_trace(go.Scatter(x=np.ones(100)*self.width/2,y= np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2))*1.05, 100), mode = "lines",line=dict(dash = "dot", color = "black", width  = 0,
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x=-np.ones(100)*self.width/2,y= np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2))*1.05, 100), mode = "lines",line=dict(dash = "dot", color = "black", width  = 0,
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x = np.array([self.width/2, self.width/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x = np.array([-self.width/2, -self.width/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))

        fig.update_layout(title={
                'text': title,
                'y':0.87,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, title_x=0.5, width=1000,
            height=600,xaxis_title= xtitle,
            yaxis_title = ytitle, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(bordercolor = "black", borderwidth = 1, x = 0.87, y = 0.7))
        fig.update_xaxes(color = "black", mirror="ticks", showline = True, visible = True, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside")
        fig.update_yaxes(color = "black", mirror="ticks", showline = True, visible = True, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside")

        fig.show()