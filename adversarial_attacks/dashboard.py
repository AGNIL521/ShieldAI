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
            html.Button('Run Image Attack', id='run-img-attack', n_clicks=0),
            html.Div(id='img-attack-result'),
            dcc.Graph(id='img-attack-plot'),
        ]),
        dcc.Tab(label='NLP Attack', children=[
            html.Button('Run NLP Attack', id='run-nlp-attack', n_clicks=0),
            html.Div(id='nlp-attack-result'),
        ]),
        dcc.Tab(label='IDS Attack', children=[
            html.Button('Run IDS Attack', id='run-ids-attack', n_clicks=0),
            html.Div(id='ids-attack-result'),
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
    Input('run-img-attack', 'n_clicks')
)
def run_img_attack(n):
    if n == 0:
        return '', go.Figure()
    # Run attack, get plot data
    data = attack_demo
    # Re-run the function to get plot data
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'simulation', 'attack_demo.py'), dtype=str, delimiter='\n')
    fooled = attack_demo.run_attack_demo(plot=False)
    # For the dashboard, show a static image (simulate)
    # Ideally, refactor attack_demo to return images, but for now, just text
    fig = go.Figure()
    fig.update_layout(title='See script output for visualization')
    return f'Adversarial attack successful? {fooled}', fig

@app.callback(
    Output('nlp-attack-result', 'children'),
    Input('run-nlp-attack', 'n_clicks')
)
def run_nlp_attack(n):
    if n == 0:
        return ''
    clean_acc, adv_acc, results = nlp_attack_demo.run_nlp_demo()
    return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}'

@app.callback(
    Output('ids-attack-result', 'children'),
    Input('run-ids-attack', 'n_clicks')
)
def run_ids_attack(n):
    if n == 0:
        return ''
    clean_acc, adv_acc = ids_attack_demo.run_ids_demo()
    return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}'

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
    app.run_server(debug=True)
