"""
Helpers module

This module contains various helper functions for PC4
"""

from typing import Callable
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import integrate

def plot_series(x: np.ndarray,
                all_t: list,
                all_U: list,
                label_u: str,
                all_Uex: list=None,
                label_uex: str=None,
                title: str="Title"
                ) -> None:
    """
    Plot a series of data and reference data with a slider
    that allows to browse through the series using plotly  

    Args:
        x (np.ndarray): Spatial grid.
        all_t (list): List of time arrays.
        all_U (list): List of data arrays.
        label_u (str): Label for the data.
        all_Uex (list): List of reference data arrays.
        label_uex (str): Label for the reference data.
        title (str): title of the plot.

    Returns:
        None
    """
    n_trace = len(all_t)

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, n_trace):
        fig.add_trace(
            go.Scatter(
                mode='markers',
                visible=False,
                line=dict(color="#00CED1", width=6),
                name=label_u,
                x=x,
                y=all_U[step]
            )
        )

    if all_Uex is not None:
        for step in np.arange(0, n_trace):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color="black", width=1),
                    name=label_uex,
                    x=x,
                    y=all_Uex[step]
                )
            )

    # Make 0th trace visible
    fig.data[0].visible = True
    if all_Uex is not None:
        fig.data[n_trace].visible = True

    if all_Uex is not None:
        fig.update_yaxes(range=[
            np.min(np.minimum(all_U,all_Uex)), 1.01*np.max(np.maximum(all_U,all_Uex))
            ])
    else:
        fig.update_yaxes(range=[
            np.min(all_U), 1.01*np.max(all_U)
            ])


    # Create and add slider
    steps = []
    for i in range(n_trace):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": f"{title} t={all_t[i]}" }],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        if all_Uex is not None:
            step["args"][0]["visible"][i+n_trace] = True  # Toggle i+n_trace'th trace to "visible"
        
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Instant: "},
        pad={"t": 30},
        steps=steps
    )]

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
    )

    fig.update_layout(
        sliders=sliders
    )

    fig.show()


def plot_series_mplt(all_t: list,
                all_U: list,
                label_u: str,
                all_Uex: list,
                label_uex: str,
                x: np.ndarray,
                title: str="Title"
                ) -> None:
    """
    Plot a series of data and reference data with a slider
    that allows to browse through the series using matplotlib  

    Args:
        all_t (list): List of time arrays.
        all_U (list): List of data arrays.
        label_u (str): Label for the data.
        all_Uex (list): List of reference data arrays.
        label_uex (str): Label for the reference data.
        x (np.ndarray): Spatial grid.
        title (str): title of the plot.

    Returns:
        None
    """
    # Generate some data


    # Create the figure and axes
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, bottom=0.25)
    plt.grid()
    plt.xlabel(r'$x$')

    # Plot the initial data
    line_u, = ax.plot(x, all_U[0], '*')
    line_uex, = ax.plot(x, all_Uex[0])

    plt.legend([line_u, line_uex], [label_u, label_uex])

    # Create the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    a_slider = Slider(ax_slider, 'Instant', 0, len(all_t)-1, valinit=0, valstep=1)

    # Update the plot when the sliders are changed
    def update(val):
        id = a_slider.val
        line_u.set_ydata(all_U[id])
        line_uex.set_ydata(all_Uex[id])
        fig.canvas.draw_idle()
    a_slider.on_changed(update)
    # Show the plot
    plt.show();

import scipy as sp
def heat_eq_exact(x: np.ndarray, t: float, u0: Callable[..., np.ndarray]) -> np.ndarray:
    """
    Computes the exact solution of the heat equation with homogeneous 
    Dirichlet boundary conditions.

    Args:
    ----
    x: np.ndarray
        The spatial grid points where the reference solution
        will be evaluated
    t: float
        the instant of time at which the reference solution
        is evaluated
    u0: Callable[..., np.ndarray]
        The initial data.

    Returns:
    -------
    uex: np.ndarray
        The exact solution at time t evaluated over all points of x.
    """

    N = 1000 # truncation in the sum evaluation
    w = np.arange(1, N+1)*np.pi*0.5
    func =  lambda y: np.sin(w*y+w)*u0(y)
    A, err = sp.integrate.quad_vec(func,-1, 1)
    E = np.exp(-t*(w**2))
    D = A*E*np.sin(w*x.reshape(-1,1)+w)
    uex = np.sum(D, axis=1)
    return uex

def burgers_eq_exact_example(x: np.ndarray, t: float, eps: float) -> np.ndarray:
    """
    This function computes the 1-periodic analytical solution for any x in (0,1)
    
    """
    r = -1.0/(4*eps*(t+1))
    uex = np.exp(r*(x-4*t)**2) + np.exp(r*(x+4*t-1)**2)
    return uex


def burgers_eq_ic_example(x: np.ndarray, eps: float) -> np.ndarray:
    """
    This function computes the 1-periodic analytical solution for any x in (0,1)
    
    """
    return burgers_eq_exact_example(x=x,t=0,eps=eps)




def elliptic_helper_F(x,f):
    x = np.atleast_1d(x)
    res = np.array([integrate.quad(f, -1, x_)[0] for x_ in x])
    return res

#%%

def elliptic_reference_solution(x,f,k):
    F_ = lambda xx: elliptic_helper_F(xx,f)
    func1 = lambda xx: - F_(xx)/k(xx)
    func2 = lambda xx: 1.0/k(xx)
    x = np.atleast_1d(x)
    I = np.array([integrate.quad(func1, -1, x_)[0] for x_ in x])
    J = np.array([integrate.quad(func2, -1, x_)[0] for x_ in x])
    C = -integrate.quad(func1, -1, 1)[0] / integrate.quad(func2, -1, 1)[0]
    res = I + C*J
    return res
