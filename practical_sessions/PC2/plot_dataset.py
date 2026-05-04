import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np

def create_plot_dataset_container(get_dataset):
    n = widgets.IntSlider(
        value=100,
        min=100,
        max=1000,
        step=10,
        description='Size n',
        continuous_update=False
    )

    eta = widgets.FloatSlider(
        value=0.,
        min=0,
        max=1.0,
        step=0.01,
        description='Perturbation',
        continuous_update=False
    )

    dataset = widgets.Dropdown(
        description='Data set',
        value='linear',
        options=['linear', 'logist', 'sphere', 'normal']
    )

    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(mode='markers', name='sampling'))

    def plot_dataset(change):
        x, y = get_dataset(dataset.value, n.value, 1, eta.value)
        fig.data[0].x = x.flat
        fig.data[0].y = y

    n.observe(plot_dataset, names='value')
    eta.observe(plot_dataset, names='value')
    dataset.observe(plot_dataset, names='value')
    plot_dataset(None)

    return widgets.VBox([dataset, n, eta, fig])

def create_plot_error_container(get_dataset, solver):
    n = widgets.IntSlider(
        value=500,
        min=1,
        max=1000,
        step=10,
        description='Size n',
        continuous_update=False
    )

    d = widgets.IntSlider(
        value=100,
        min=1,
        max=200,
        step=10,
        description='Dimension d',
        continuous_update=False
    )

    eta = widgets.FloatRangeSlider(
        value=[0., 0.5],
        min=0,
        max=1.0,
        step=0.01,
        description='Perturbation',
        continuous_update=False
    )

    dataset = widgets.Dropdown(
        description='Data set',
        value='linear',
        options=['linear', 'logist', 'sphere', 'normal']
    )

    fig = go.FigureWidget()
    fig.update_xaxes(title_text='data perturbation')
    fig.update_yaxes(title_text='error')
    fig.add_trace(go.Bar())

    def plot_error(change):
        eta_array = np.linspace(eta.value[0], eta.value[1], 10)
        error = []
        for e in eta_array:
            x, y = get_dataset(dataset.value, n.value, d.value, e)
            A = np.column_stack([x])
            w = solver(A, y)
            error.append(np.square(x @ w - y).mean())
        fig.data[0].x = eta_array
        fig.data[0].y = error

    n.observe(plot_error, names='value')
    d.observe(plot_error, names='value')
    eta.observe(plot_error, names='value')
    dataset.observe(plot_error, names='value')
    plot_error(None)

    return widgets.VBox([dataset, n, d, eta, fig])