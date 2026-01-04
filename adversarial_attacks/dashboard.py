import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import sys, os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath('simulation'))
from simulation import attack_demo, nlp_attack_demo, ids_attack_demo, defense_adversarial_training, defense_input_randomization, monitoring_example

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Adversarial Attacks Dashboard'

app.layout = dbc.Container([
    html.H1("Adversarial Attacks & Defenses Dashboard", className="text-center my-4"),
    dbc.Tabs([
    dbc.Tab(label='Image Attack', children=[
            dbc.Card([
                dbc.CardBody([
                    html.Label('Dataset:', className="fw-bold"),
                    dcc.Dropdown(
                        id='img-dataset',
                        options=[
                            {'label': 'Digits (sklearn)', 'value': 'digits'},
                            {'label': 'MNIST', 'value': 'mnist'},
                            {'label': 'Fashion-MNIST', 'value': 'fashion-mnist'},
                            {'label': 'CIFAR-10', 'value': 'cifar10'},
                            {'label': 'Upload Image Dataset', 'value': 'upload'}
                        ],
                        value='digits',
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Upload(
                        id='img-upload',
                        children=html.Div(['Drag and Drop or ', html.A('Select Files'), html.Br(),
                                        'Supported: .csv, .tsv, .npy, .npz, .txt, .xls, .xlsx, .json, .zip (ZIP: preview only)']),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'margin': '10px 0'},
                        className="mb-3"
                    ),
                    html.Div(id='img-upload-preview', style={'marginBottom': '10px', 'fontSize': 'small', 'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'}),
                    html.Label('Attack Strength (epsilon):', className="fw-bold"),
                    dcc.Slider(id='img-epsilon', min=0, max=1, step=0.05, value=0.3, marks={0:'0', 0.5:'0.5', 1:'1'}, className="mb-3"),
                    dcc.Checklist(id='img-random', options=[{'label': ' Randomize Sample', 'value': 'random'}], value=['random'], className="mb-3"),
                    dbc.Button('Run Image Attack', id='run-img-attack', n_clicks=0, color="primary", className="mb-3"),
                    dbc.Alert(id='img-attack-result', color="info", is_open=True),
                    dcc.Graph(id='img-attack-plot'),
                ])
            ], className="p-3")
    ]),
    dbc.Tab(label='NLP Attack', children=[
            dbc.Card([
                dbc.CardBody([
                    html.Label('Dataset:', className="fw-bold"),
                    dcc.Dropdown(
                        id='nlp-dataset',
                        options=[
                            {'label': 'Toy Spam/Ham', 'value': 'toy'},
                            {'label': 'SMS Spam Collection', 'value': 'sms'},
                            {'label': '20 Newsgroups', 'value': '20news'},
                            {'label': 'Upload Text Dataset', 'value': 'upload'}
                        ],
                        value='toy',
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Upload(
                        id='nlp-upload',
                        children=html.Div(['Drag and Drop or ', html.A('Select Files'), html.Br(),
                                        'Supported: .csv, .tsv, .txt, .xls, .xlsx, .json']),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'margin': '10px 0'},
                        className="mb-3"
                    ),
                    html.Div(id='nlp-upload-preview', style={'marginBottom': '10px', 'fontSize': 'small', 'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'}),
                    html.Label('Perturbation Probability:', className="fw-bold"),
                    dcc.Slider(id='nlp-perturb-prob', min=0, max=1, step=0.05, value=0.3, marks={0:'0', 0.5:'0.5', 1:'1'}, className="mb-3"),
                    dcc.Checklist(id='nlp-random', options=[{'label': ' Randomize Test Order', 'value': 'random'}], value=['random'], className="mb-3"),
                    dbc.Button('Run NLP Attack', id='run-nlp-attack', n_clicks=0, color="primary", className="mb-3"),
                    dbc.Alert(id='nlp-attack-result', color="info"),
                    dcc.Graph(id='nlp-attack-bar'),
                ])
            ], className="p-3")
        ]),
        dbc.Tab(label='IDS Attack', children=[
            dbc.Card([
                dbc.CardBody([
                    html.Label('Dataset:', className="fw-bold"),
                    dcc.Dropdown(
                        id='ids-dataset',
                        options=[
                            {'label': 'Toy', 'value': 'toy'},
                            {'label': 'NSL-KDD', 'value': 'nslkdd'},
                            {'label': 'UNSW-NB15', 'value': 'unsw'},
                            {'label': 'Upload IDS Dataset', 'value': 'upload'}
                        ],
                        value='toy',
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Upload(
                        id='ids-upload',
                        children=html.Div(['Drag and Drop or ', html.A('Select Files'), html.Br(),
                                        'Supported: .csv, .tsv, .txt, .xls, .xlsx, .json']),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'margin': '10px 0'},
                        className="mb-3"
                    ),
                    html.Div(id='ids-upload-preview', style={'marginBottom': '10px', 'fontSize': 'small', 'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'}),
                    html.Label('Attack Strength (epsilon):', className="fw-bold"),
                    dcc.Slider(id='ids-epsilon', min=0, max=5, step=0.1, value=1.0, marks={0:'0', 2.5:'2.5', 5:'5'}, className="mb-3"),
                    dbc.Button('Run IDS Attack', id='run-ids-attack', n_clicks=0, color="primary", className="mb-3"),
                    dbc.Alert(id='ids-attack-result', color="info"),
                    dcc.Graph(id='ids-attack-bar'),
                ])
            ], className="p-3")
        ]),
        dbc.Tab(label='Adversarial Training', children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Button('Run Adversarial Training', id='run-adv-train', n_clicks=0, color="success", className="mb-3"),
                    dbc.Alert(id='adv-train-result', color="success"),
                ])
            ], className="p-3")
        ]),
        dbc.Tab(label='Input Randomization', children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Button('Run Input Randomization', id='run-input-rand', n_clicks=0, color="warning", className="mb-3"),
                    dbc.Alert(id='input-rand-result', color="warning"),
                ])
            ], className="p-3")
        ]),
        dbc.Tab(label='Monitoring', children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Button('Run Monitoring', id='run-monitoring', n_clicks=0, color="danger", className="mb-3"),
                    dbc.Alert(id='monitoring-result', color="danger"),
                ])
            ], className="p-3")
        ]),
    ], className="mb-3")
], fluid=True)

@app.callback(
    Output('img-attack-result', 'children'),
    Output('img-attack-plot', 'figure'),
    Input('run-img-attack', 'n_clicks'),
    State('img-epsilon', 'value'),
    State('img-random', 'value'),
    State('img-dataset', 'value'),
    State('img-upload', 'contents')
)
def run_img_attack(n, epsilon, randomize, dataset, upload_contents):
    if n == 0:
        return 'Ready to run.', go.Figure()
    logger.info(f"Running Image Attack: dataset={dataset}, epsilon={epsilon}")
    idx = None if 'random' in (randomize or []) else 0
    # Pass dataset and upload_contents to backend (to be implemented in attack_demo.py)
    try:
        fooled, orig, adv, diff = attack_demo.run_attack_demo(plot=False, return_images=True, epsilon=epsilon, idx=idx, dataset=dataset, upload_contents=upload_contents)
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
    except Exception as e:
        logger.error(f"Error in Image Attack: {e}", exc_info=True)
        return f"Error: {str(e)}", go.Figure()

@app.callback(
    Output('nlp-attack-result', 'children'),
    Output('nlp-attack-bar', 'figure'),
    Input('run-nlp-attack', 'n_clicks'),
    State('nlp-perturb-prob', 'value'),
    State('nlp-random', 'value'),
    State('nlp-dataset', 'value'),
    State('nlp-upload', 'contents')
)
def run_nlp_attack(n, perturb_prob, randomize, dataset, upload_contents):
    if n == 0:
        return 'Ready to run.', go.Figure()
    logger.info(f"Running NLP Attack: dataset={dataset}, prob={perturb_prob}")
    # If randomize, shuffle test samples by seeding np.random
    import numpy as np
    if 'random' in (randomize or []):
        np.random.seed(None)
    # Pass dataset and upload_contents to backend (to be implemented in nlp_attack_demo.py)
    try:
        clean_acc, adv_acc, results = nlp_attack_demo.run_nlp_demo(perturb_prob=perturb_prob, dataset=dataset, upload_contents=upload_contents)
        fig = go.Figure(data=[
            go.Bar(name='Clean', x=['Accuracy'], y=[clean_acc]),
            go.Bar(name='Adversarial', x=['Accuracy'], y=[adv_acc])
        ])
        fig.update_layout(barmode='group', title='NLP Attack: Clean vs Adversarial Accuracy')
        return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}', fig
    except Exception as e:
        logger.error(f"Error in NLP Attack: {e}", exc_info=True)
        return f"Error: {str(e)}", go.Figure()

@app.callback(
    Output('ids-attack-result', 'children'),
    Output('ids-attack-bar', 'figure'),
    Input('run-ids-attack', 'n_clicks'),
    State('ids-epsilon', 'value'),
    State('ids-dataset', 'value'),
    State('ids-upload', 'contents')
)
def run_ids_attack(n, epsilon, dataset, upload_contents):
    if n == 0:
        return 'Ready to run.', go.Figure()
    logger.info(f"Running IDS Attack: dataset={dataset}, epsilon={epsilon}")
    try:
        clean_acc, adv_acc = ids_attack_demo.run_ids_demo(epsilon=epsilon, dataset=dataset, upload_contents=upload_contents)
        fig = go.Figure(data=[
            go.Bar(name='Clean', x=['Accuracy'], y=[clean_acc]),
            go.Bar(name='Adversarial', x=['Accuracy'], y=[adv_acc])
        ])
        fig.update_layout(barmode='group', title='IDS Attack: Clean vs Adversarial Accuracy')
        return f'Clean accuracy: {clean_acc:.2f}, Adversarial accuracy: {adv_acc:.2f}', fig
    except Exception as e:
        logger.error(f"Error in IDS Attack: {e}", exc_info=True)
        return f"Error: {str(e)}", go.Figure()

@app.callback(
    Output('adv-train-result', 'children'),
    Input('run-adv-train', 'n_clicks')
)
def run_adv_train(n):
    if n == 0:
        return 'Ready to run.'
    logger.info("Running Adversarial Training")
    try:
        clean_acc, adv_acc = defense_adversarial_training.run_adversarial_training_demo()
        return f'Adversarially trained accuracy on clean: {clean_acc:.2f}, on adversarial: {adv_acc:.2f}'
    except Exception as e:
        logger.error(f"Error in Adversarial Training: {e}", exc_info=True)
        return f"Error: {str(e)}"

@app.callback(
    Output('input-rand-result', 'children'),
    Input('run-input-rand', 'n_clicks')
)
def run_input_rand(n):
    if n == 0:
        return 'Ready to run.'
    logger.info("Running Input Randomization")
    try:
        acc = defense_input_randomization.run_input_randomization_demo()
        return f'Accuracy on adversarial+randomized inputs: {acc:.2f}'
    except Exception as e:
        logger.error(f"Error in Input Randomization: {e}", exc_info=True)
        return f"Error: {str(e)}"

@app.callback(
    Output('monitoring-result', 'children'),
    Input('run-monitoring', 'n_clicks')
)
def run_monitoring(n):
    if n == 0:
        return 'Ready to run.'
    logger.info("Running Monitoring")
    try:
        flags = monitoring_example.run_monitoring_demo()
        return f'Potential adversarial samples at indices: {flags}'
    except Exception as e:
        logger.error(f"Error in Monitoring: {e}", exc_info=True)
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
