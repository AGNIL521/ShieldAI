import dash
from dash import html, dcc, Output, Input, State
import plotly.graph_objs as go
import numpy as np
import sys, os
sys.path.append(os.path.abspath('simulation'))
from simulation import attack_demo, nlp_attack_demo, ids_attack_demo, defense_adversarial_training, defense_input_randomization, monitoring_example

app = dash.Dash(__name__)
app.title = 'Adversarial Attacks Dashboard'

app.layout = html.Div([
    html.H1("Adversarial Attacks & Defenses Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Image Attack', children=[
            html.Label('Attack Strength (epsilon):'),
            dcc.Slider(id='img-epsilon', min=0, max=1, step=0.05, value=0.3, marks={0:'0', 0.5:'0.5', 1:'1'}),
            html.Button('Run Image Attack', id='run-img-attack', n_clicks=0),
            html.Div(id='img-attack-result'),
            dcc.Graph(id='img-attack-plot'),
        ]),
        dcc.Tab(label='NLP Attack', children=[
            html.Label('Perturbation Probability:'),
            dcc.Slider(id='nlp-perturb-prob', min=0, max=1, step=0.05, value=0.3, marks={0:'0', 0.5:'0.5', 1:'1'}),
            html.Button('Run NLP Attack', id='run-nlp-attack', n_clicks=0),
            html.Div(id='nlp-attack-result'),
            dcc.Graph(id='nlp-attack-bar'),
        ]),
        dcc.Tab(label='IDS Attack', children=[
            html.Label('Attack Strength (epsilon):'),
            dcc.Slider(id='ids-epsilon', min=0, max=5, step=0.1, value=1.0, marks={0:'0', 2.5:'2.5', 5:'5'}),
            html.Button('Run IDS Attack', id='run-ids-attack', n_clicks=0),
            html.Div(id='ids-attack-result'),
            dcc.Graph(id='ids-attack-bar'),
        ]),
        dcc.Tab(label='Adversarial Training', children=[
            html.Button('Run Adversarial Training', id='run-adv-train', n_clicks=0),
            html.Div(id='adv-train-result'),
        ]),
        dcc.Tab(label='Input Randomization', children=[
            html.Button('Run Input Randomization', id='run-input-rand', n_clicks=0),
            html.Div(id='input-rand-result'),
        ]),
        dcc.Tab(label='Monitoring', children=[
            html.Button('Run Monitoring', id='run-monitoring', n_clicks=0),
            html.Div(id='monitoring-result'),
        ]),
    ])
])

@app.callback(
    Output('img-attack-result', 'children'),
    Output('img-attack-plot', 'figure'),
    Input('run-img-attack', 'n_clicks'),
    State('img-epsilon', 'value')
)
def run_img_attack(n, epsilon):
    if n == 0:
        return '', go.Figure()
    fooled, orig, adv, diff = attack_demo.run_attack_demo(plot=False, return_images=True, epsilon=epsilon)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=orig, colorscale='gray', showscale=False, name='Original'))
    fig.add_trace(go.Heatmap(z=adv, colorscale='gray', showscale=False, name='Adversarial', visible=False))
    fig.add_trace(go.Heatmap(z=diff, colorscale='rdylbu', showscale=False, name='Difference', visible=False))
    fig.update_layout(
        title='Original, Adversarial, and Difference Images',
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {'label': 'Original', 'method': 'update', 'args': [{'visible': [True, False, False]}, {'title': 'Original Image'}]},
                {'label': 'Adversarial', 'method': 'update', 'args': [{'visible': [False, True, False]}, {'title': 'Adversarial Image'}]},
                {'label': 'Difference', 'method': 'update', 'args': [{'visible': [False, False, True]}, {'title': 'Difference Image'}]},
            ]
        }]
    )
    return f'Adversarial attack successful? {fooled}', fig

@app.callback(
    Output('nlp-attack-result', 'children'),
    Output('nlp-attack-bar', 'figure'),
    Input('run-nlp-attack', 'n_clicks'),
    State('nlp-perturb-prob', 'value')
)
def run_nlp_attack(n, perturb_prob):
    if n == 0:
        return '', go.Figure()
    clean_acc, adv_acc, results = nlp_attack_demo.run_nlp_demo(perturb_prob=perturb_prob)
    fig = go.Figure(data=[
        go.Bar(name='Clean', x=['Accuracy'], y=[clean_acc]),
        go.Bar(name='Adversarial', x=['Accuracy'], y=[adv_acc])
    ])
    fig.update_layout(barmode='group', title='NLP Attack: Clean vs Adversarial Accuracy')
    return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}', fig

@app.callback(
    Output('ids-attack-result', 'children'),
    Output('ids-attack-bar', 'figure'),
    Input('run-ids-attack', 'n_clicks'),
    State('ids-epsilon', 'value')
)
def run_ids_attack(n, epsilon):
    if n == 0:
        return '', go.Figure()
    clean_acc, adv_acc = ids_attack_demo.run_ids_demo(epsilon=epsilon)
    fig = go.Figure(data=[
        go.Bar(name='Clean', x=['Accuracy'], y=[clean_acc]),
        go.Bar(name='Adversarial', x=['Accuracy'], y=[adv_acc])
    ])
    fig.update_layout(barmode='group', title='IDS Attack: Clean vs Adversarial Accuracy')
    return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}', fig

@app.callback(
    Output('adv-train-result', 'children'),
    Input('run-adv-train', 'n_clicks')
)
def run_adv_train(n):
    if n == 0:
        return ''
    clean_acc, adv_acc = defense_adversarial_training.run_adversarial_training_demo()
    return f'Adversarially trained accuracy on clean: {clean_acc:.2f}, on adversarial: {adv_acc:.2f}'

@app.callback(
    Output('input-rand-result', 'children'),
    Input('run-input-rand', 'n_clicks')
)
def run_input_rand(n):
    if n == 0:
        return ''
    acc = defense_input_randomization.run_input_randomization_demo()
    return f'Accuracy on adversarial+randomized inputs: {acc:.2f}'

@app.callback(
    Output('monitoring-result', 'children'),
    Input('run-monitoring', 'n_clicks')
)
def run_monitoring(n):
    if n == 0:
        return ''
    flags = monitoring_example.run_monitoring_demo()
    return f'Potential adversarial samples at indices: {flags}'

if __name__ == '__main__':
    app.run(debug=True)
