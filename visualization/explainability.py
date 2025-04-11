import plotly.graph_objects as go
import shap
import numpy as np
import pandas as pd
from typing import List, Optional

def plot_shap_summary(model, X: pd.DataFrame, feature_names: Optional[List[str]] = None, max_display: int = 20, title='SHAP Feature Importance') -> go.Figure:
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    if feature_names is None:
        feature_names = X.columns.tolist()

    vals = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(zip(feature_names, vals), columns=['Feature', 'SHAP Value'])
    feature_importance.sort_values(by='SHAP Value', ascending=False, inplace=True)

    fig = go.Figure(go.Bar(
        x=feature_importance['SHAP Value'][:max_display],
        y=feature_importance['Feature'][:max_display],
        orientation='h',
        marker_color='#1f77b4'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        yaxis={'categoryorder': 'total ascending'},
        height=600,
        margin=dict(l=150)
    )

    return fig


def plot_shap_dependence(model, X: pd.DataFrame, feature: str, interaction_feature: Optional[str] = None, title='SHAP Dependence Plot') -> go.Figure:
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap_feature_values = shap_values[:, X.columns.get_loc(feature)].values

    fig = go.Figure()

    if interaction_feature:
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=shap_feature_values,
            mode='markers',
            marker=dict(
                color=X[interaction_feature],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=interaction_feature)
            ),
            name=f'{feature} vs {interaction_feature}'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=shap_feature_values,
            mode='markers',
            name=feature
        ))

    fig.update_layout(
        title=title,
        xaxis_title=feature,
        yaxis_title='SHAP Value'
    )

    return fig
