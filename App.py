import streamlit as st
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
import os

def pace_to_float(t):
    if pd.isna(t) or str(t).strip() in ('', '--'):
        return np.nan
    parts = str(t).strip().split(':')
    try:
        if len(parts) == 2:  # mm:ss
            return int(parts[0]) + int(parts[1])/60
        elif len(parts) == 3:  # hh:mm:ss
            return int(parts[0])*60 + int(parts[1]) + int(parts[2])/60
    except:
        return np.nan
    return np.nan


#1 Load the data
df_weight = pd.read_excel('Weight_20250928.xlsx')
df_weight['time'] = pd.to_datetime(df_weight['time'])

# Find relative max/min indexes
n = 3 # window width for extrema
rel_max = argrelextrema(df_weight['Body weight(kg)'].values, np.greater, order=n)
rel_min = argrelextrema(df_weight['Body weight(kg)'].values, np.less, order=n)

df_G = pd.read_csv('Activities_Run.csv')
df_G['Date'] = pd.to_datetime(df_G['Date'], errors='coerce')
df_G['Distance'] = pd.to_numeric(df_G['Distance'], errors='coerce')
df_G['month'] = df_G['Date'].dt.to_period('M')
df_monthly = df_G.groupby('month')['Distance'].sum().reset_index()
df_monthly['month'] = df_monthly['month'].dt.to_timestamp()

df_G['Avg Pace'] = df_G['Avg Pace'].apply(pace_to_float)
#df_G['Avg Run Cadence'] = df_G['Avg Run Cadence'].apply(pace_to_float)
df_avg_pacing = df_G.groupby(['month'])['Avg Pace'].mean().reset_index()
df_avg_pacing['month'] = df_avg_pacing['month'].dt.to_timestamp()

df_G['run_type'] = pd.cut(df_G['Distance'], bins=[0, 7.5, 12.5, 17.5, np.inf], labels=['5km', '10km', '15km', '20km+'])

#2 main program
tab1, tab2, tab3 = st.tabs(['Weight', 'Distance per month', 'Pacing vs Cadence'])

with tab1:  #Weight
    st.title('Weight vs Date')
    fig = go.Figure()
    # Main weight line
    fig.add_trace(go.Scatter(
        x=df_weight['time'], y=df_weight['Body weight(kg)'], mode='lines+markers', name='Weight',
        marker=dict(size=2, color='gray')
    ))
    # Relative maxima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_max], y=df_weight['Body weight(kg)'].iloc[rel_max], mode='markers', name='Rel max',
        marker=dict(size=4, color='red', symbol='diamond'),
        hovertemplate='<b>Max</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'
    ))

    # Relative minima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_min], y=df_weight['Body weight(kg)'].iloc[rel_min], mode='markers', name='Rel min',
        marker=dict(size=4, color='blue', symbol='star'),
        hovertemplate='<b>Min</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'
    ))

    fig.update_layout(
        title="Weight vs Date with Range Slider and Clickable Max/Min",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=12, label="12m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis_title="Weight (kg)",
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2: #Distance per month
    st.title('Distance per Month')
     # Get min and max dates
    min_date = df_monthly['month'].min().date()
    max_date = df_monthly['month'].max().date()

    # Range slider for selecting period
    start_date, end_date = st.slider(
        "Drag to select period:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )

    # Filter by slider range
    mask = (df_monthly['month'].dt.date >= start_date) & (df_monthly['month'].dt.date <= end_date)
    df_filtered = df_monthly.loc[mask]

    # Calculate stats
    total_distance = df_filtered['Distance'].sum()
    total_months = df_filtered['month'].nunique()

    # Show dynamic metrics
    st.metric("Total Distance (km)", f"{total_distance:.1f}")
    years = total_months // 12
    months = total_months % 12
    if years > 0:
        st.metric("Period", f"{years} year{'s' if years > 1 else ''} {months} month{'s' if months!= 1 else ''}")
    else:
        st.metric("Period", f"{months} month{'s' if months!= 1 else ''}")


    # Plot filtered chart
    fig = px.bar(
        df_filtered, x='month', y='Distance',
        title='Distance per Month',
        labels={'month': 'Month', 'Distance': 'Distance (km)'},
        text_auto='.0f'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3: #Pacing vs Cadence
    st.title('Pacing vs Cadence')
    #st.write(df_G[['run_type', 'Avg Pace', 'Avg Run Cadence']])
    color_map = {'5km': 'blue', '10km': 'red', '15km': 'green', '20km+': 'purple'}
    fig = go.Figure()
    for run in color_map.keys():
        df_sub = df_G[df_G['run_type'] == run]
        #st.write(f"Run type {run} has {len(df_sub)} rows")
        #st.write(f"Run type {run} has {len(df_sub)} rows")
        #st.write(df_sub[['Avg Pace', 'Avg Run Cadence']])
        fig.add_trace(go.Scatter(
            x=df_sub['Avg Pace'], y=df_sub['Avg Run Cadence'], mode='markers',
            marker=dict(color=color_map[run]), name=run,
            hovertemplate='Pacing: %{x:.2f} min/km<br>Cadence: %{y:.1f} spm<br>Distance: '+run
        ))
    fig.update_layout(title='Pacing vs Cadence by Run Distance',
                        xaxis_title='Pacing (min/km)', yaxis_title='Cadence (steps/min)')
    st.plotly_chart(fig, use_container_width=True)
