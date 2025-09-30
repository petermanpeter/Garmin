import streamlit as st
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
import os

#0 Target file to be imported
df_weight = pd.read_excel('Weight_20250928.xlsx')
df_G = pd.read_csv('Activities_Run_20250928.csv')

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
df_weight['time'] = pd.to_datetime(df_weight['time'])

# Find relative max/min indexes
n = 3 # window width for extrema
rel_max = argrelextrema(df_weight['Body weight(kg)'].values, np.greater, order=n)
rel_min = argrelextrema(df_weight['Body weight(kg)'].values, np.less, order=n)

df_G['Date'] = pd.to_datetime(df_G['Date'], errors='coerce')
df_G['Distance'] = pd.to_numeric(df_G['Distance'], errors='coerce')
df_G['month'] = df_G['Date'].dt.to_period('M')
df_monthly = df_G.groupby('month')['Distance'].sum().reset_index()
df_monthly['month'] = df_monthly['month'].dt.to_timestamp()

df_G['Avg Pace ori'] = df_G['Avg Pace']
df_G['Avg Pace'] = df_G['Avg Pace'].apply(pace_to_float)

df_avg_pacing = df_G.groupby(['month'])['Avg Pace'].mean().reset_index()
df_avg_pacing['month'] = df_avg_pacing['month'].dt.to_timestamp()

df_G['run_type'] = pd.cut(df_G['Distance'], bins=[0, 7.5, 12.5, 17.5, np.inf], labels=['5km', '10km', '15km', '20km+'])

#2 main program
#tab1, tab2, tab3, tab4 = st.tabs(['Weight', 'Distance per month', 'Pacing vs Cadence', 'GarminConnect login'])
tab1, tab2, tab3, tab4 = st.tabs(['Weight', 'Distance per month', 'Pacing vs Cadence', 'GarminConnect login'])

with tab1:  #Weight
    st.title('Weight vs Date')
    unit = 'lb'
    unit = st.radio("Choose unit:", ['lb', 'kg'], index=0)  # default kg
    if unit == 'kg':
        weight = df_weight['Body weight(kg)']
        yaxis_title = "Weight (kg)"
    else:
        weight = df_weight['Body weight(kg)'] * 2.20462  # convert to lb
        yaxis_title = "Weight (lb)"
    fig = go.Figure()
    # Main weight line
    fig.add_trace(go.Scatter(
        x=df_weight['time'], y=weight, mode='lines+markers', name='Weight',
        marker=dict(size=1, color='gray')
    ))
    # Relative maxima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_max], y=weight.iloc[rel_max], mode='markers', name='Rel max',
        marker=dict(size=2, color='red', symbol='diamond'),
        hovertemplate='<b>Max</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'+unit
    ))

    # Relative minima
    fig.add_trace(go.Scatter(
        x=df_weight['time'].iloc[rel_min], y=weight.iloc[rel_min], mode='markers', name='Rel min',
        marker=dict(size=2, color='blue', symbol='star'),
        hovertemplate='<b>Min</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'+unit
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
        yaxis_title=yaxis_title,
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
    df_filtered['Distance_rounded'] = df_filtered['Distance'].round(0)

    # Show dynamic metrics
    st.metric("Total Distance (km)", f"{total_distance:,.1f}")
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
        labels={'month': 'Month', 'Distance_rounded': 'Distance (km)'},
        text_auto='.0f',
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3: #Pacing vs Cadence
    st.title('Pacing vs Cadence')
    #st.write(df_G[['run_type', 'Avg Pace', 'Avg Run Cadence']])
    run_types = ['5km', '10km', '15km', '20km+']
    selected = st.multiselect('Select Run Types:', run_types, default=run_types)
    color_map = {'5km': 'blue', '10km': 'red', '15km': 'green', '20km+': 'purple'}
    fig = go.Figure()
    df_filtered = df_G[df_G['run_type'].isin(selected)]
    #for run in color_map.keys():
    for run in selected:
        #df_sub = df_G[df_G['run_type'] == run]
        df_sub = df_filtered[df_filtered['run_type'] == run]
        #st.write(f"Run type {run} has {len(df_sub)} rows")
        #st.write(df_sub[['Avg Pace', 'Avg Run Cadence']])
        #Title = df_sub[['Title']].values
        #Time = df_sub[['Time']].values
        #Distance = df_sub[['Distance']].values
        fig.add_trace(go.Scatter(
            x=df_sub['Avg Pace'], y=df_sub['Avg Run Cadence'], mode='markers',
            marker=dict(color=color_map[run]), name=run,
            customdata=df_sub[['Title', 'Time', 'Distance','Avg Pace ori']].values,
            hovertemplate=(
            '%{customdata[0]}<br>Time: %{customdata[1]}<br>Distance: ('+run+') %{customdata[2]} km<br>'
            'Pacing: %{customdata[3]} min/km<br>Cadence: %{y:.0f} spm'
            )
            #hovertemplate='Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm<br>Distance: '+run
            #hovertemplate=Title_val +'<br>Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm<br>Distance: '+run
            #hovertemplate=Title +'<br>Time: ' + Time +'<br>Distance: ('+run+') <br>Pacing: %{x:.2f} min/km<br>Cadence: %{y:.0f} spm'
            #hovertemplate=Title + '<br>Time: ' +Time+'<br>Distance: ('+run+')'+str(Distance)+' <br>Pacing: ' + str(Avg_Pace) + ' min/km<br>Cadence: %{y:.0f} spm'
        ))
    fig.update_layout(title='Pacing vs Cadence by Run Distance',
                        xaxis_title='Pacing (min/km)', yaxis_title='Cadence (steps/min)')
    st.plotly_chart(fig, use_container_width=True)

with tab4:  #GarminConnect login
    st.title('GarminConnect login (under development)')
    #main()