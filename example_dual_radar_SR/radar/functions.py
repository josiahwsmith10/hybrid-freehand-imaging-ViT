import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
pyo.init_notebook_mode()

def plot_image_2d(im, x_m, y_m, xyz_str="xy", title="Reconstructed Image"):
    surface = [go.Surface(z=im, x=y_m, y=x_m)]

    fig = go.Figure(data=surface)
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    
    fig.update_layout(title=title, 
                    autosize=True,
                    scene = dict(
                    xaxis_title=f"{xyz_str[0]} (m)",
                    yaxis_title=f"{xyz_str[1]} (m)",
                    zaxis_title="Reflectivity",
                    xaxis = dict(
                        nticks=10),
                    yaxis = dict(
                        nticks=10),
                    zaxis = dict(
                        nticks=10),),
                    width=700,
                    margin=dict(l=65, r=50, b=65, t=90),
                  )

    fig.show()
    
def plot_image_2d_dB(im, x_m, y_m, xyz_str="xy", title="Reconstructed Image"):
    im = im / im.max()
    im = 10*np.log10(im)
    
    surface = [go.Surface(z=im, x=y_m, y=x_m)]

    fig = go.Figure(data=surface)
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    
    fig.update_layout(title=title, 
                    autosize=True,
                    scene = dict(
                    xaxis_title=f"{xyz_str[0]} (m)",
                    yaxis_title=f"{xyz_str[1]} (m)",
                    zaxis_title="Reflectivity",
                    xaxis = dict(
                        nticks=10),
                    yaxis = dict(
                        nticks=10),
                    zaxis = dict(
                        nticks=10),),
                    width=700,
                    margin=dict(l=65, r=50, b=65, t=90),
                  )

    fig.show()
    