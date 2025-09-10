from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
import os

def pace_to_float(t):
    import numpy as np
    if pd.isna(t) or str(t).strip() in ('', '--'):
        return np.nan
    parts = str(t).strip().split(':')
    try:
        if len(parts) == 2:  # mm:ss
            return int(parts) + int(parts) / 60
        elif len(parts) == 3:  # hh:mm:ss
            return int(parts) * 60 + int(parts) + int(parts) / 60
    except:
        return np.nan
    return np.nan


#2a Tab1 (Weight)
# Load the data
df_weight = pd.read_excel('Weight_20250907.xlsx')

# Convert Date column to datetime
df_weight['time'] = pd.to_datetime(df_weight['time'])

# Plot by day
#plt.figure(figsize=(10, 5))
#plt.plot(df['time'], df['Body weight(kg)'], marker='o')
#plt.title('Body Weight (kg) vs Day')
#plt.xlabel('Date')
#plt.ylabel('Body Weight (kg)')
#plt.grid(True)
#plt.tight_layout()
#plt.show()

# Find relative max/min indexes
n = 3 # window width for extrema
rel_max = argrelextrema(df_weight['Body weight(kg)'].values, np.greater, order=n)
rel_min = argrelextrema(df_weight['Body weight(kg)'].values, np.less, order=n)

#3a Tab2 (Distance run by month)
df_G = pd.read_csv('Activities_Run.csv')
df_G['Date'] = pd.to_datetime(df_G['Date'], errors='coerce')
df_G['Distance'] = pd.to_numeric(df_G['Distance'], errors='coerce')
df_G['month'] = df_G['Date'].dt.to_period('M')
df_monthly = df_G.groupby('month')['Distance'].sum().reset_index()
df_monthly['month'] = df_monthly['month'].dt.to_timestamp()

df_G['Avg Pace'] = df_G['Avg Pace'].apply(pace_to_float)
df_avg_pacing = df_G.groupby(['month'])['Avg Pace'].mean().reset_index()
df_avg_pacing['month'] = df_avg_pacing['month'].dt.to_timestamp()

#4a Tab3 (Pacing vs Cadence)
run_type = pd.cut(df_G['Distance'], bins=[0, 7.5, 12.5, 17.5, np.inf], labels=['5km', '10km', '15km', '20km+'])

#
app = Dash(__name__)

# Example datasets for demonstration
df2 = px.data.gapminder().query("year==2007")
df3 = px.data.tips()
df4 = px.data.carshare()
df5 = px.data.medals_wide()

#0 Python Dash tab layout
app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Weight', value='tab-1'),
        dcc.Tab(label='Distance per month', value='tab-2'),
        dcc.Tab(label='Pacing vs Cadence', value='tab-3'),
        dcc.Tab(label='Avg pacing by run type', value='tab-4'),
        dcc.Tab(label='Medals Area', value='tab-5'),
    ]),
    html.Div(id='tabs-content-example')
])
@app.callback(
    Output('tabs-content-example', 'children'),
    Input('tabs-example', 'value')
)


def render_content(tab):
    if tab == 'tab-1':
#2b Tab1 (Weight)
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

    elif tab == 'tab-2':
#3b Tab2 (Distance run by month)
        fig = px.bar(df_monthly, x='month', y='Distance',
                     title='Distance Traveled per Month',
                     labels={'month':'Month', 'Distance':'Distance (km)'})
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))
        return dcc.Graph(figure=fig)
    
    elif tab == 'tab-3':
#4b Tab3 (Pacing vs Cadence)
        color_map = {'5km': 'blue', '10km': 'red', '15km': 'green', '20km+': 'purple'}
        fig = go.Figure()
        for run in color_map.keys():
            df_sub = df_G[df_G['run_type'] == run]
            fig.add_trace(go.Scatter(
                x=df_sub['Avg pace'], y=df_sub['Avg Run Cadence'], mode='markers',
                marker=dict(color=color_map[run]), name=run,
                hovertemplate='Pacing: %{x:.2f} min/km<br>Cadence: %{y:.1f} spm<br>Distance: '+run
            ))
        fig.update_layout(title='Pacing vs Cadence by Run Distance',
                          xaxis_title='Pacing (min/km)', yaxis_title='Cadence (steps/min)')
        return dcc.Graph(figure=fig)

    elif tab == 'tab-4':
        fig = px.line(df4, x='centroid_lon', y='centroid_lat',
                      title="Carshare Locations")
    elif tab == 'tab-5':
        fig = px.area(df5, x='nation', y=['gold', 'silver', 'bronze'],
                      title="Medal Counts by Country")
    else:
        fig = {}

    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    # app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8050)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8501)))

