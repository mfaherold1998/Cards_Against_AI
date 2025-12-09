import plotly.express as px
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import plotly.graph_objects as go

def plot_toxicity_vs_temperature (df:pd.DataFrame):
    
    df_plot_line = df.groupby(['model', 'temperature'], as_index=False).agg(
        mean_tox = ('toxicity', 'mean'),
        std_tox = ('toxicity', 'std')
    )
    df_plot_line['std_tox'] = df_plot_line['std_tox'].fillna(0)

    df_plot_line['upper_bound'] = df_plot_line['mean_tox'] + df_plot_line['std_tox']
    max_y_value = df_plot_line['upper_bound'].max()
    y_max_with_margin = max_y_value * 1.05
    y_max_final = min(1.0, y_max_with_margin)

    fig = px.line(
        data_frame=df_plot_line,
        x='temperature',
        y='mean_tox',
        color='model',
        markers=True,
        custom_data=['std_tox'],
        title='Mean Toxicity vs Temperature (per model)',
        labels={
            'mean_tox': 'Mean Toxicity (Score)'
        }
    )

    fig.update_traces(
        error_y=dict(
            type='data',
            array=df_plot_line['std_tox'],      # El tamaño positivo del error
            arrayminus=df_plot_line['std_tox'] # El tamaño negativo del error
        )
    )

    hover_template = '<b>Mean Toxicity (Score):</b> %{y:.5f}<br>' + \
                     '<b>Std Dev:</b> ±%{customdata[0]:.5f}' + \
                     '<extra></extra>'
    
    fig.update_traces(
        hoverinfo='text',
        hovertemplate=hover_template
         
    )

    fig.update_layout(
        xaxis_title='Temperature',
        yaxis_title='Mean Toxicity',
        # Ajustar el rango del eje Y si es necesario (scores de 0 a 1)
        yaxis_range=[0, y_max_final],
        width=800,
        xaxis_range=[0.4, 0.9]
    )

    fig.update_xaxes(
        dtick=0.3,  # Establece el intervalo entre ticks a 0.10
        tick0=0.5,
        tickformat=".1f" # Asegura que se muestre con un decimal
    )

    return fig

def plot_toxicity_vs_temperature_shaded(df: pd.DataFrame):
    
    g = (df.groupby(['model', 'temperature'])['toxicity']
           .agg(["mean", "count", "std"]).reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    
    # Calculate the limits for the 95% confidence interval (mean ± 1.96*weeks)
    g["upper_bound"] = g["mean"] + 1.96 * g["sem"].fillna(0)
    g["lower_bound"] = g["mean"] - 1.96 * g["sem"].fillna(0)

    min_data = g["lower_bound"].min()
    max_data = g["upper_bound"].max()
    data_range = max_data - min_data
    margin = data_range * 0.05 # 5% de margen
    
    min_val = min_data - margin
    max_val = max_data + margin

    fig = px.line(
        g,
        x='temperature', 
        y='mean', 
        color='model', 
        line_group='model',
        markers=True,
        title=f"Mean toxicity vs Temperature (shaded area)",
        labels={
            'mean': "Mean Toxicity",
            'temperature': "Temperature"
        },
        hover_data={
            'model': False,
            'temperature': False,
            "std": ":.4f",
            "count": True,
            "sem": ":.4f",
            "mean": ":.4f"
        }
    )
    
    fig.update_yaxes(range=[min_val, max_val])

    models = g['model'].unique()

    fig.update_xaxes(
        dtick=0.3,  # Establece el intervalo entre ticks a 0.10
        tick0=0.5,
        tickformat=".1f" # Asegura que se muestre con un decimal
    )

    for i, m in enumerate(models):
        sub = g[g['model'] == m].sort_values('temperature')
        
        try:
            line_color = fig.data[i].line.color
        except IndexError:
            print(f"Warning: Color not found for model {m}. Using gray.")
            line_color = 'rgba(128, 128, 128, 0.2)'
    
        # Add the shaded area (error band)
        fig.add_trace(go.Scatter(
            x=pd.concat([sub['temperature'], sub['temperature'].sort_values(ascending=False)]),
            y=pd.concat([sub["upper_bound"], sub["lower_bound"].sort_values(ascending=False)]),
            fill='toself',
            fillcolor=line_color.replace('rgb', 'rgba').replace(')', ', 0.2)'), 
            line_color='rgba(255,255,255,0)',
            name=f'{m} (95% CI)',
            showlegend=False,
            hoverinfo='skip',
            opacity=0.2
        ))

    return fig

def plot_distribution_by_model (df:pd.DataFrame):
    
    fig = px.violin(
        data_frame=df,
        x='model',
        y='toxicity',
        color='model',
        box=True,
        points='all',
        hover_data=['lang', 'temperature'],
        title='Toxicity Distribution per model',
        labels={
            'model': 'LLM',
            'toxicity': 'Toxicity (Score)'
        },
        template="plotly_white"
    )

    fig.update_traces(
        
        selector=dict(mode='markers'),
        hoverinfo='text'
    )
    
    hover_template = '<b>Modelo:</b> %{x}<br>' + \
                    '<b>Toxicidad:</b> %{y:.4f}<br>' + \
                    '<b>Idioma:</b> %{customdata[0]}<br>' + \
                    '<b>Temperatura:</b> %{customdata[1]}<extra></extra>'

    fig.update_traces(
        hovertemplate=hover_template,
        selector=dict(type='violin') 
    )

    return fig

