import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = '202503_fire_stats.csv'
df = pd.read_csv(file_path)

# Clean numeric columns
df['Number of Fires'] = df['Number of Fires'].astype(str).str.replace(',', '').astype(float)
df['Burned Area'] = df['Burned Area'].astype(str).str.replace(',', '').astype(float)

# Sort by Year
df = df.sort_values(by='Year')

# Focus on the last 25 years (from 2000 to 2025)
df_recent = df[df['Year'] >= 2000].copy()

# Function to calculate Central Line and Process Limits
def calculate_xmr_limits(series, baseline_size=5):
    central_line = series.iloc[:baseline_size].mean()
    moving_ranges = series.diff().abs().iloc[1:baseline_size]
    avg_mr = moving_ranges.mean()
    lpl = central_line - (2.66 * avg_mr)
    upl = central_line + (2.66 * avg_mr)
    return central_line, avg_mr, lpl, upl

# Plotting with regression and XmR lines
def plot_with_regression_and_xmr(x, y, x_label, y_label, title, filename, xmr_x, xmr_y):
    x = x.values.reshape(-1, 1)
    y = y.values

    # Linear Regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # XmR Calculations
    cl, avg_mr, lpl, upl = calculate_xmr_limits(xmr_y)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(x, y, 'o', label='Actual Data', alpha=0.6)
    plt.plot(x, y_pred, 'r-', label='Best Fit Line', linewidth=2)

    # Highlight XmR section
    plt.plot(xmr_x, xmr_y, 'bo-', label='XmR Section (2000–2025)', linewidth=2)

    # Add XmR lines
    plt.axhline(cl, color='green', linestyle='--', label='Central Line (XmR)')
    plt.axhline(upl, color='orange', linestyle='--', label='Upper Natural Process Limit')
    plt.axhline(lpl, color='orange', linestyle='--', label='Lower Natural Process Limit')

    # Final touches
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot 1: Number of Fires
plot_with_regression_and_xmr(
    df['Year'],
    df['Number of Fires'],
    x_label='Year',
    y_label='Number of Fires',
    title='Year vs Number of Fires (with XmR Chart)',
    filename='fires_xmr_chart.png',
    xmr_x=df_recent['Year'].values,
    xmr_y=df_recent['Number of Fires']
)

# Plot 2: Burned Area
plot_with_regression_and_xmr(
    df['Year'],
    df['Burned Area'],
    x_label='Year',
    y_label='Burned Area',
    title='Year vs Burned Area (with XmR Chart)',
    filename='burned_area_xmr_chart.png',
    xmr_x=df_recent['Year'].values,
    xmr_y=df_recent['Burned Area']
)


