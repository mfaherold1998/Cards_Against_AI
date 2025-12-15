import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.utils import build_play_key

ATTRIBUTE_COLUMNS = ['toxicity', 'severe_toxicity', 'obcene', 'threat', 'insult', 'identity_attack', 'sexually_explicit', 'profanity']

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

def plot_rates_above_threshold (df:pd.DataFrame, thr:float = 0.5, column:str = 'toxicity'):
    
    tmp = df.copy()
    tmp["tail"] = (tmp[column] >= thr).astype(int) 

    # Percentage of toxicity above the threshold.
    rate = (tmp.groupby(['model', 'temperature'])["tail"]
              .mean().mul(100).reset_index())
    
    fig = px.line(
        rate,
        x='temperature', 
        y="tail", 
        color='model', 
        markers=True,
        title="High Tail by Model",
        labels={
            "tail": f"% of Toxicity ≥ {thr}",
            'temperature': "Temperature",
            'model': "Model"
        },
        hover_data={
            'model': False,
            'temperature': True,
            "tail": ":.2f"
        }
    )
    
    fig.update_xaxes(
        dtick=0.3,
        tick0=0.5,
        tickformat=".1f"
    ) 
    fig.update_yaxes(range=[0, rate['tail'].max() * 1.1]) 
    
    return fig

def plot_black_card_triggers (df: pd.DataFrame, top_k: int = 10, column:str = 'toxicity'):
    
    tmp = df.copy()
    
    top_black = (tmp.groupby("black_id")[column]
                    .mean().sort_values(ascending=False).head(top_k).index)

    mat = (tmp[tmp["black_id"].isin(top_black)]
            .pivot_table(index="black_id", columns='model',
                         values='toxicity', aggfunc="mean"))
    
    if mat.empty:
        return None

    fig = px.imshow(
        mat,
        x=mat.columns,
        y=mat.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title=f"Top {top_k} Toxic Black Cards (Mean per Model)",
        labels={"x": "Model", "y": "Black Card ID", "color": f"Mean Tpxicity"},
        text_auto=".2f", 
        height=top_k * 25 + 150
    )
    
    fig.update_coloraxes(colorbar_title="Mean Toxicity")
    fig.update_yaxes(tickangle=0)

    fig.update_traces(
        hovertemplate=(
            "<b>Mean Toxicity:</b> %{z:.4f}<extra></extra>"
        )
    )
    
    return fig

def plot_top_plays_heatmap_per_model(df: pd.DataFrame, top_k: int = 10, column:str = 'toxicity'):
    
    tmp = df.copy()
    
    tmp["play_key"] = [build_play_key(b, w) for b, w in zip(tmp['black_id'], tmp['winners'])]

    all_models = tmp['model'].unique()
    num_models = len(all_models)

    if num_models == 0 or tmp.empty:
        return None
    
    fig = make_subplots(
        rows=num_models, 
        cols=1,
        subplot_titles=[f"Top {top_k} Toxic Plays: Model {m}" for m in all_models],
        vertical_spacing=0.05
    )

    row_counter = 1

    for model_name in all_models:
            
        df_model = tmp[tmp['model'] == model_name].copy()
       
        top_plays_series = (df_model.groupby("play_key")[column]
                            .mean().sort_values(ascending=False).head(top_k))
        
        top_plays_index = top_plays_series.index
        
        if top_plays_index.empty:
            continue

        mat = (df_model[df_model["play_key"].isin(top_plays_index)]
                .pivot_table(index="play_key", columns='model',
                            values=column, aggfunc="mean"))
        
        mat = mat.reindex(top_plays_index)

        heatmap_trace = go.Heatmap(
            z=mat.values,
            x=mat.columns,
            y=mat.index,
            colorscale='Viridis',
            name=model_name,
            showscale=True,
            text=mat.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate=(
                f"<b>Mean {column.title()}:</b> %{{z:.4f}}<br>"
            )
        )

        fig.add_trace(heatmap_trace, row=row_counter, col=1)
        fig.update_yaxes(
            row=row_counter, col=1, 
            autorange="reversed", 
            tickangle=0, 
            tickfont=dict(size=10)
        )

        if row_counter < num_models:
            fig.update_xaxes(row=row_counter, col=1, showticklabels=False)
            
        row_counter += 1

    fig.update_layout(
        title_text=f"Top {top_k} Toxic Plays per Model — Ranked by {column.title()}",
        height=num_models * top_k * 30 + 150,
        width=800
    )

    for i in range(1, num_models + 1):
        if i > 1:           
            fig.data[i-1].showscale = False
        else:
            fig.data[i-1].colorbar = dict(
                title=dict(
                    text=f"Mean {column.title()}",
                    side="right"
                ),
                len=0.9,
                y=0.5
            )

    return fig

def plot_instability(df: pd.DataFrame, top_n: int = 10):
    
    tmp = df.copy()
    tmp["play_key"] = [build_play_key(b, w) for b, w in zip(tmp['black_id'], tmp['winners'])]

    stability = (tmp.groupby(['model', 'temperature', "play_key"])['toxicity']
                   .agg(["mean", "std", "count"]).reset_index()
                   .rename(columns={"mean": "tox_mean", "std": "tox_std", "count": "n"}))
    
    if stability.empty:
        return go.Figure().add_annotation(text="No data to plot.")

    # 2. Graph 1: Scatterplot (Toxicity vs. Instability)
    stability["size_px"] = stability["n"] * 500 / max(stability["n"].max(), 1)
    
    fig_scatter_px = px.scatter(
        stability, 
        x="tox_mean", 
        y="tox_std",
        color='model', 
        size="size_px", 
        hover_data=['play_key', 'n', 'temperature'],
        title="Mean Toxicity vs. Instability (Standard Deviation)",
        labels={
            "tox_mean": "Average Toxicity (tox_mean)",
            "tox_std": "Instability (tox_std)",
            "size_px": "N Observations"
        }
    )

    # 3. Graph 2: Barplot (Top N Instability)
    top_play_keys = (stability.groupby("play_key")["tox_std"]
                     .mean().sort_values(ascending=False).head(top_n).index)
    
    top_unstable = stability[stability["play_key"].isin(top_play_keys)].copy()
    
    order_map = top_unstable.groupby('play_key')['tox_std'].mean().sort_values(ascending=False)
    
    fig_bar_px = px.bar(
        top_unstable, 
        x="tox_std", 
        y="play_key", 
        color='model',
        orientation='h',
        title=f"Top {top_n} Most Unstable Combinations (High STD)",
        category_orders={"play_key": order_map.index.tolist()},
        labels={
            "tox_std": "STD of Toxicity (tox_std)",
            "play_key": "Play Key (Black | White)"
        }
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Scatter: Mean Toxicity vs. Instability",
            f"Bar: Top {top_n} Unstable Plays"
        ),
        row_heights=[0.6, 0.4]
    )
    
    for trace in fig_scatter_px.data:
        fig.add_trace(go.Scatter(trace), row=1, col=1)

    for trace in fig_bar_px.data:
        fig.add_trace(go.Bar(trace), row=2, col=1)
    
    fig.update_traces(marker_line_width=0, selector=dict(type='bar'))
    fig.update_traces(marker_line_width=0, selector=dict(type='scatter'))
    
    fig.update_layout(
        title_text=f"Instability Analysis by Model",
        height=800,
        showlegend=True,
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', row=2, col=1)
    
    return fig

def plot_category_comparison(df: pd.DataFrame):
    
    present_attributes = [col for col in ATTRIBUTE_COLUMNS if col in df.columns]

    if not present_attributes:
        return None

    agg = df.groupby('model')[present_attributes].mean().reset_index()

    df_melted = agg.melt(
        id_vars='model',
        value_vars=present_attributes,
        var_name='Attribute',
        value_name='Average_Score'
    )
    
    if df_melted.empty:
        return None

    fig = px.bar(
        df_melted,
        x='model',
        y='Average_Score',
        color='Attribute',
        barmode='group',
        title=f"Profile of Attributes per Model",
        labels={
            "model": "Model",
            "Average_Score": "Average Score per Attribute",
            "Attribute": "Attribute Category"
        },
        height=600
    )
    
    fig.update_xaxes(tickangle=30)
    
    fig.update_traces(
        hovertemplate=(
            "<b>Average Score:</b> %{y:.4f}<extra></extra>"
        )
    )
    
    return fig

def _summarize_toxicity(d: pd.DataFrame) -> pd.DataFrame:
    return (d.groupby(['model', 'lang'])['toxicity'] 
            .agg(mean="mean",
                 p50=lambda s: s.quantile(.5),
                 p90=lambda s: s.quantile(.9),
                 p95=lambda s: s.quantile(.95))
            .round(3)
            .reset_index()) 

def plot_language_risk_faceted(df: pd.DataFrame):

    df_clean = df.dropna(subset=['lang']).copy()
    
    if df_clean.empty:
        return None

    agg_df = _summarize_toxicity(df_clean)
    
    df_melted = agg_df.melt(
        id_vars=['model', 'lang'],
        value_vars=["mean", "p50", "p90", "p95"],
        var_name='Metric',
        value_name='Score'
    )

    fig = px.bar(
        df_melted,
        x='model',
        y='Score',
        color='Metric',
        facet_col='lang',
        barmode='group',
        title=f"Toxicity Profile per Model, Faceted by Language",
        labels={
            'model': "Model",
            "Score": "Toxicity Score",
            "Metric": "Metric"
        },
        height=550,
        #width=250 * len(df_clean['lang'].unique()) + 150 
        width=900
    )
    
    fig.update_yaxes(range=[0, 1.05], title_text="Score")
    fig.update_xaxes(tickangle=30)
    
    fig.update_layout(
        margin=dict(l=50, r=50, t=80, b=50),
        legend_title_text="Métricas de Riesgo",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
        
    return fig

def plot_config_toxicity_per_model(df: pd.DataFrame, min_n: int = 1, column:str = 'toxicity'):
    
    CLEAN_CONFIG = 'clean_config'
    tmp = df.copy()

    if 'config' not in tmp.columns:
        return None
        
    tmp[CLEAN_CONFIG] = tmp['config'].astype(str)

    stats = (tmp.groupby([CLEAN_CONFIG, 'model'])[column]
               .agg(mean="mean", count="count").reset_index())
    
    stats = stats[stats["count"] >= min_n]
    
    if stats.empty:
        return None

    mat = stats.pivot(index=CLEAN_CONFIG, columns='model', values="mean")
    
    if mat.empty:
        return None

    fig = px.imshow(
        mat,
        x=mat.columns,
        y=mat.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title=f"Mean Toxicity by Configuration per Model",
        labels={"x": "Model", "y": "Configuration", "color": "Mean Toxicity"},
        text_auto=".3f",
        height=max(500, 30 * len(mat.index) + 100),
        width=800
    )
    
    fig.update_coloraxes(colorbar_title="Mean Toxicity")

    fig.update_traces(
        hovertemplate="<extra></extra>"
        )
    

    return fig

def plot_config_distribution(df: pd.DataFrame, target_config:str = 'random_games_5'):
    
    CLEAN_CONFIG = 'clean_config'
    tmp = df.copy()

    # 1. Limpiar el nombre de la configuración
    if 'config' not in tmp.columns:
        return None
        
    tmp[CLEAN_CONFIG] = tmp['config'].astype(str)

    df_filtered = tmp[tmp[CLEAN_CONFIG] == target_config].copy()
    
    if df_filtered.empty:
        return None

    fig = px.violin(
        df_filtered,
        x='toxicity',
        y=CLEAN_CONFIG,
        color='model',
        orientation='h',
        box=True,
        points='outliers',
        # category_orders={'CLEAN_CONFIG': sorted(tmp[CLEAN_CONFIG].unique())},
        title=f"Toxicity Distribution by Configuration (Split by Model)",
        labels={
            'toxicity': "Toxicity Score",
            CLEAN_CONFIG: "Game Configuration"
        },
        height=max(600, 30 * tmp[CLEAN_CONFIG].nunique() + 100), 
        width=900
    )
    
    fig.update_xaxes(range=[0, 1.05], title_text=f"Toxicity Score")
    
    fig.update_layout(
        legend_title_text="Model",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.05
        )
    )
    
    return fig

def plot_config_tail_rate_(df: pd.DataFrame, thr: float = 0.20, column:str = 'toxicity'):
    
    CLEAN_CONFIG = 'clean_config'
    tmp = df.copy()

    if 'config' not in tmp.columns:
        return None
    
    tmp[CLEAN_CONFIG] = tmp['config'].astype(str)
    tmp["tail"] = (tmp[column] >= thr).astype(int)

    rate = (tmp.groupby([CLEAN_CONFIG, 'model'])["tail"]
              .mean().mul(100).reset_index())
    
    if rate.empty:
        return None

    fig = px.bar(
        rate,
        x=CLEAN_CONFIG,
        y="tail",
        color='model',
        barmode='group',
        title=f"High-Toxicity Rate (% ≥ {thr}) by Configuration",
        labels={
            CLEAN_CONFIG: "Configuration",
            "tail": f"% of Scores ≥ {thr}",
            'model': "Model"
        },
        height=600,
        width=900 
    )
    
    fig.update_xaxes(
        tickangle=30, 
        title_text="Configuration"
    )
    fig.update_yaxes(
        range=[0, 100], 
        title_text=f"% of Scores ≥ {thr}"
    )

    fig.update_layout(
        legend_title_text="Model",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

