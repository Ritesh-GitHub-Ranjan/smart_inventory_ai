import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Tuple

def plot_forecast_with_ci(df: pd.DataFrame, date_col='Date', actual_col='Sales Quantity', forecast_col='prediction', lower_col='pred_lower', upper_col='pred_upper') -> go.Figure:
    """
    Creates an interactive line chart with forecast and confidence intervals
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], name='Forecast', line=dict(color='green', dash='dot')))
    
    ci_x = pd.concat([df[date_col], df[date_col][::-1]])
    ci_y = pd.concat([df[upper_col], df[lower_col][::-1]])

    fig.add_trace(go.Scatter(
        x=ci_x,
        y=ci_y,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Demand Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Demand',
        hovermode='x unified'
    )
    
    return fig


def plot_seasonal_decomposition(df: pd.DataFrame, date_col='Date', value_col='Sales Quantity', period=7) -> Tuple[go.Figure, dict]:
    """
    Creates seasonal decomposition plots (trend, seasonal, residual)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(df.set_index(date_col)[value_col], period=period)

    fig = make_subplots(rows=4, cols=1, subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])

    fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.observed, name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df[date_col], y=decomposition.resid, name='Residual'), row=4, col=1)

    fig.update_layout(height=800, title_text="Seasonal Decomposition", showlegend=False)

    return fig, {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }
