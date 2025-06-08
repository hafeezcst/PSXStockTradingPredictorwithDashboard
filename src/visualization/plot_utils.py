"""
Reusable plotting utilities for PSX dashboard and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def plot_pie_chart(ax, data, title, colors, explode=None):
    """Reusable pie chart plotting function with improved aesthetics."""
    if explode is None:
        explode = [0.05] * len(data)
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct='%.2f%%',
        colors=colors, explode=explode, shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

def plot_bar_chart(ax, metrics, values, colors, title, ylabel):
    """Reusable bar chart plotting function with value annotations."""
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', fontsize=10)
    return bars

def draw_trend_lines(ax, df, column, color, label):
    """Draw trend lines for a given column."""
    try:
        x = np.arange(len(df))
        y = df[column].values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            return
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        coeffs = np.polyfit(x_valid, y_valid, 1)
        trend_line = np.polyval(coeffs, x)
        ax.plot(df['Date'], trend_line, color=color, linestyle='--', label=label, alpha=0.7)
    except Exception as e:
        print(f"Error drawing trend for {label}: {str(e)}") 