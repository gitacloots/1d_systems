###
# This document contains all the visualisation tools for rapid debugging and specific information
import tools
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def show_1d_quantity(self, quantity, title = "Title", xaxis = "x", yaxis = "y"):
        fig = go.Figure()
        xtitle = xaxis

        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(quantity), name = r"Real part", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.imag(quantity), name = r"Imaginary part", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

def show_1d(self, prop, z, d, w, i, title, name_1, name_2, yaxis):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        y1 = np.real(prop[w, i, :])
        y2 = np.imag(prop[w, i, :])
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y1, name = name_1, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y2, name = name_2, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        
        fig.add_annotation(
            x=tools.center_z(z)[i],
            y=np.real(prop[w, i, i]),
            text=r'$z_1$',
            showarrow=True,
            xanchor="right",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=50,
            ay=-30,
            font=dict(
            size=16,
            ),
        )
        fig.add_trace(go.Scatter(x = np.array([d/2, d/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        fig.add_trace(go.Scatter(x = np.array([-d/2, -d/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        fig.show()

def show_1d_omega(prop, omega, z, i, title, name_1, name_2, yaxis):
        fig = go.Figure()
        xtitle = r"$\omega \text{ eV}$"
        prop_diag = tools.extract_diag(prop)
        fig.add_trace(go.Scatter(x = omega*27.211, y = np.real(prop_diag[:, i]), name = name_1, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = omega*27.211, y = np.imag(prop_diag[:, i]), name = name_2, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_annotation(
            xref="paper", 
            yref="paper",
            x=0.8, 
            y=0.9,
            text="z = "+str(tools.center_z(z)[i])+" Bohr",
            showarrow=False,
            font=dict(color = "black",
            size=13,
            ),
        )
        
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        fig.show()

def show_map(prop, z, omega, width, nz, nz_slab, w, title):
    fig = go.Figure(data =
    go.Contour(x = tools.center_z(z), y = tools.center_z(z),
        z=prop[w], name = "Freq = "+str(omega[w]*27.211)+" eV"
    ))
    fig.update_layout(title_text = title, title_x=0.5,xaxis_title= r'$z_1$', height = 800, width = 800, 
            yaxis_title = r'$z_2$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = True)
    fig.add_trace(go.Scatter(x = np.ones(nz_slab)*width/2, y = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False))
    fig.add_trace(go.Scatter(x = -np.ones(nz_slab)*width/2, y = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_trace(go.Scatter(y = np.ones(nz_slab)*width/2, x = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_trace(go.Scatter(y = -np.ones(nz_slab)*width/2, x = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_annotation(
        x=z[round(nz*0.25)],
        y=z[round(nz*0.25)],
        text="Freq = "+str(omega[w]*27.211)+" eV",
        showarrow=False,
        font=dict(color = "black",
        size=16,
        ),
    )
    fig.show()

def show_map_omega(prop, z, omega, width, nz, title):
    ytitle = r"$\omega \text{ [eV]}$"
    prop_diag = np.real(tools.extract_diag(prop))
    fig = go.Figure(data =
    go.Contour(x = tools.center_z(z), y = omega*27.211,
        z=prop_diag,
    ))
    fig.update_layout(title_text = title, title_x=0.5,xaxis_title= r'$z$',
            yaxis_title = ytitle,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = False)
    fig.update_traces(colorscale="Deep", selector=dict(type='contour'))
    fig.add_trace(go.Scatter(x = np.ones(nz)*width/2, y = omega*27.211, mode = "lines", line=dict(color = "black", dash =  'dash'),))
    fig.add_trace(go.Scatter(x = -np.ones(nz)*width/2, y = omega*27.211, mode = "lines", line=dict(color = "black", dash =  'dash'),))
    fig.show()


def show_spat_mode(dict_in, lg_choice, nws_choice, index_x, index_y, x_info):
    if x_info == 'qp':
        d_space = np.array([0])
        for i in dict_in.keys():
            if i == 'info':
                info = dict_in[i]
            else:
                d_space = np.append(d_space, dict_in[i]['normal']['qp'])
    elif x_info == 'space':
        d_space = np.array([])
        for i in dict_in.keys():
            if i == 'info':
                info = dict_in[i]
            else:
                d_space = np.append(d_space, dict_in[i]['normal']['space'])
    #index_x = np.argwhere(index_x == d_space)[0, 0]
    z = dict_in[index_x][nws_choice]['z_vec']
    if lg_choice=='Surface function':
        lg_choice = 'g'
    elif lg_choice=='Loss function':
        lg_choice = 'l'
    vec = dict_in[index_x][nws_choice]['response_functions'][lg_choice]['modes'][index_y]['vec']
    vec_dual = dict_in[index_x][nws_choice]['response_functions'][lg_choice]['modes'][index_y]['vec_dual']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    xtitle = r'$\huge{z [\mathring{\text{A}}]}$'
    ytitle = r'$\Huge{V_i(z), \rho_i(z)}$'
    #title = r'$\text{Eigenmodes in the EELS spectra}$'
    y1 = np.real(np.fft.fft(vec, norm = 'ortho'))
    y2 = np.real(np.fft.ifft(vec_dual, norm = 'ortho'))
    fig.add_trace(go.Scatter(x = tools.center_z(z)/1.8897259886, y = y1*3 , showlegend = False, name = r'$V_i(z)$', mode = "lines",marker=dict(
        size=5,color = "blue", 
        ),),secondary_y=False,)
    fig.add_trace(go.Scatter(x = tools.center_z(z)/1.8897259886, y =  y2, showlegend = False, name = r'$\rho_i(z)$', mode = "lines",marker=dict(
        size=5,color = "red",
        ),),secondary_y=True,)
    width = info['width_one_slab']/1.8897259886
    space = d_space[index_x]/1.8897259886
    y_low = min(min(y1), min(y2))
    y_high = max(max(y1), max(y2))
    
    if x_info == 'qp':
        fig.add_trace(go.Scatter(x = np.array([width/2, width/2]), y =  np.array([y_low, y_high])*1.05, name = "Surfaces", mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x = np.array([-width/2, -width/2]), y =  np.array([y_low, y_high])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
                color = "black",
                ), showlegend = False))
    elif x_info == 'space':
        fig.add_trace(go.Scatter(x = np.array([space/2, space/2]), y =  np.array([y_low, y_high])*1.05, name = "Surfaces", mode = "lines",line=dict(dash = "dot",
                color = "black",
                ), showlegend = False))

        fig.add_trace(go.Scatter(x = np.array([-space/2, -space/2]), y =  np.array([y_low, y_high])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
                color = "black",
                ), showlegend = False))

        fig.add_trace(go.Scatter(x = np.array([space/2+width, space/2+width]), y =  np.array([y_low, y_high])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
                color = "black",
                ), showlegend = False))

        fig.add_trace(go.Scatter(x = np.array([-space/2-width, -space/2-width]), y =  np.array([y_low, y_high])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
                color = "black",
                ), showlegend = False))
    fig.update_layout(xaxis_title= xtitle, height = 600,
        yaxis_title = ytitle, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(bordercolor = "black", borderwidth = 1, x = 0.75, y = 0.93))
    fig.update_xaxes(color = "black", showline = False, visible = True, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside", range = [-35, 35], tickvals = [-30, -15, 0, 15, 30])
    fig.update_yaxes(title_text=r'$\huge{V_i(z)}$', color = "black", showline = True, visible = False, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside", secondary_y=False)
    fig.update_yaxes(title_text=r'$\huge{\rho_i(z)}$', color = "black", showline = True, visible = False, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside", secondary_y=True)
    fig.update_layout(xaxis = dict(tickfont = dict(size = 20), ticklen = 20), yaxis = dict(tickfont = dict(size = 20), ticklen = 20), font=dict(size=15))

    return fig