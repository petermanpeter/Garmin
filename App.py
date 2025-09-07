from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import argrelextrema

app = Dash(__name__)

# Example datasets for demonstration
df1 = px.data.iris()
df2 = px.data.gapminder().query("year==2007")
df3 = px.data.tips()
df4 = px.data.carshare()
df5 = px.data.medals_wide()

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Weight', value='tab-1'),
        dcc.Tab(label='Gapminder Bar', value='tab-2'),
        dcc.Tab(label='Tips Pie', value='tab-3'),
        dcc.Tab(label='Carshare Line', value='tab-4'),
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
        # Load the data
        df = pd.read_excel('Weight_20250907.xlsx')

        # Convert Date column to datetime
        df['time'] = pd.to_datetime(df['time'])

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
        rel_max = argrelextrema(df['Body weight(kg)'].values, np.greater, order=n)
        rel_min = argrelextrema(df['Body weight(kg)'].values, np.less, order=n)

        fig = go.Figure()

        # Main weight line
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Body weight(kg)'], mode='lines+markers', name='Weight',
            marker=dict(size=2, color='gray')
        ))

        # Relative maxima
        fig.add_trace(go.Scatter(
            x=df['time'].iloc[rel_max], y=df['Body weight(kg)'].iloc[rel_max], mode='markers', name='Rel max',
            marker=dict(size=4, color='red', symbol='diamond'),
            hovertemplate='<b>Max</b><br>Date: %{x|%Y-%m-%d}<br>Weight: %{y:.1f}'
        ))

        # Relative minima
        fig.add_trace(go.Scatter(
            x=df['time'].iloc[rel_min], y=df['Body weight(kg)'].iloc[rel_min], mode='markers', name='Rel min',
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
        fig = px.bar(df2, x='continent', y='pop', color='continent',
                     title="Gapminder 2007 Population by Continent")
    elif tab == 'tab-3':
        fig = px.pie(df3, values='total_bill', names='day',
                     title="Tips Total Bill by Day")
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
   # app.run(debug=True, host='0.0.0.0', port=8050)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8050)
