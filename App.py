import streamlit as st
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
#from __future__ import print_function
import os
from io import BytesIO
import base64
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from garminconnect import Garmin, GarminConnectAuthenticationError
import json

#0 Target file to be imported
df_G = pd.read_csv('Activities_Run_20251202.csv')
garmin_file = 'Weight_20251202.xlsx'

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

# ---------------- CONFIG ----------------
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SEARCH_SENDER = "PICOOC"
SEARCH_SUBJECT = "Health Data file"
SAVE_DIR = "."
# ----------------------------------------


def gmail_authenticate():
    """Authenticate Gmail using OAuth (requires credentials.json)."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            #creds = flow.run_local_server(port=0)
            #creds = flow.run_console()
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("Please go to this URL: ", auth_url)
            code = input("Enter the authorization code here: ")
            creds = flow.fetch_token(code=code)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def search_latest_message(service, query):
    """Find the most recent Gmail message matching a query."""
    results = service.users().messages().list(userId="me", q=query, maxResults=1).execute()
    messages = results.get("messages", [])
    if not messages:
        return None
    msg_id = messages[0]["id"]
    return service.users().messages().get(userId="me", id=msg_id).execute()


def get_attachment_as_bytes(service, message):
    """Return (filename, bytes, received_date_str) for the Excel attachment."""
    headers = message.get("payload", {}).get("headers", [])
    date_header = next((h["value"] for h in headers if h["name"] == "Date"), None)
    if date_header:
        try:
            received_dt = datetime.datetime.strptime(date_header[:25], "%a, %d %b %Y %H:%M:%S")
        except Exception:
            received_dt = datetime.datetime.utcnow()
    else:
        received_dt = datetime.datetime.utcnow()

    date_str = received_dt.strftime("%Y%m%d")

    parts = message.get("payload", {}).get("parts", [])
    if not parts:
        parts = [message.get("payload", {})]

    for part in parts:
        filename = part.get("filename")
        if not filename:
            continue
        if filename.lower().endswith(".xlsx"):
            att_id = part["body"].get("attachmentId")
            if not att_id:
                continue
            att = service.users().messages().attachments().get(
                userId="me", messageId=message["id"], id=att_id
            ).execute()
            file_data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))
            return filename, file_data, date_str
    return None, None, date_str


#1 Load the data
#service = gmail_authenticate()
#query = f'from:{SEARCH_SENDER} subject:"{SEARCH_SUBJECT}"'
#message = search_latest_message(service, query)
#if not message:
#    print("❌ No matching PICOOC email found.")
#filename, file_bytes, date_str = get_attachment_as_bytes(service, message)
#if not file_bytes:
#    print("❌ No .xlsx attachment found.")

# Save locally
#save_name = f"Weight_{date_str}.xlsx"
#with open(save_name, "wb") as f:
#    f.write(file_bytes)
#print(f"✅ Saved attachment as {save_name}")

# ---- NEW PART: Load into Pandas ----
try:
    df_weight = pd.read_excel(garmin_file)
    #df_weight = pd.read_excel(BytesIO(file_bytes))
    #print(f"✅ Loaded Excel into DataFrame ({len(df)} rows, {len(df.columns)} columns)")
    #print(df.head())  # preview first few rows
except Exception as e:
    print("⚠️ Failed to read Excel:", e)

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
    #save_attachment(service, message)
    #df_weight = pd.read_excel('Weight_20250928.xlsx')
    df_weight['time'] = pd.to_datetime(df_weight['time'])
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
    mask = (df_monthly['month'].dt.to_period('M') >= pd.Period(start_date, freq='M')) & (df_monthly['month'].dt.to_period('M') <= pd.Period(end_date, freq='M'))
    #mask = (df_monthly['month'].dt.date >= start_date) & (df_monthly['month'].dt.date <= end_date)
    df_filtered = df_monthly.loc[mask]

    # Calculate stats
    total_distance = df_filtered['Distance'].sum()
    filtered_yyyymm_min = pd.Period(start_date, freq='M')
    filtered_yyyymm_max = pd.Period(end_date, freq='M')
    total_months = (filtered_yyyymm_max.year - filtered_yyyymm_min.year) * 12 + (filtered_yyyymm_max.month - filtered_yyyymm_min.month) + 1
    #total_months = df_filtered['month'].nunique()
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
    st.title('GarminConnect login')
    email = st.text_input("Garmin email")
    password = st.text_input("Garmin password", type="password")

    if st.button("Login & fetch runs"):
        if not email or not password:
            st.error("Please enter email and password")
        else:
            try:
                api = Garmin(email, password)
                api.login()

                today = datetime.date.today()
                start_date = today - datetime.timedelta(days=365)

                # Get activities in last 30 days (Garmin API uses offset + limit)
                # Simple approach: fetch first N recent activities then filter by date and type
                raw_acts = api.get_activities(0, 200)  # adjust limit if needed

                acts = []
                for a in raw_acts:
                    act_date = datetime.datetime.strptime(a["startTimeLocal"][:10], "%Y-%m-%d").date()
                    if act_date < start_date:
                        continue
                    if a.get("activityType", {}).get("typeKey") not in ["running", "trail_running"]:
                        continue
                    acts.append(a)

                if not acts:
                    st.info("No run activities found in the last 30 days.")
                else:
                    df = pd.DataFrame(acts)
                    st.write("Found runs:", len(df))
                    df["distance_km"] = df["distance"] / 1000.0
                    df["distance_km"] = df["distance_km"].round(2)
                    st.dataframe(df[["activityId", "activityName", "startTimeLocal", "distance_km"]])

                    # Select a run to show map
                    sel_id = st.selectbox(
                        "Select a run to show map",
                        df["activityId"].tolist(),
                        format_func=lambda x: f"{x} - " +
                        df.loc[df["activityId"] == x, "activityName"].iloc[0]
                    )

                    if sel_id:
                        details = api.get_activity_details(sel_id)
                        #st.json(details.get("geoPolylineDTO", {}))
                        # GPS samples are in detail data; path may differ per version
                        # Common structure: "geoPolylineDTO" or "activityDetailMetrics"[...]
                        geo = details.get("geoPolylineDTO")
                        if not geo:
                            st.warning("No GPS polyline available for this activity.")
                        else:
                            # Points as list of [lat, lon]
                            pts = [(p["lat"], p["lon"]) for p in geo.get("polyline", [])]
                            if not pts:
                                st.warning("No GPS track points found.")
                            else:
                                gps_df = pd.DataFrame(pts, columns=["lat", "lon"])
                                fig = px.line_mapbox(
                                    gps_df,
                                    lat="lat",
                                    lon="lon",
                                    zoom=12,
                                    height=500,
                                )
                                fig.update_layout(
                                    mapbox_style="open-street-map",
                                        mapbox=dict(
                                        zoom=13,
                                        center=dict(lat=gps_df["lat"].mean(), lon=gps_df["lon"].mean()),
                                    ),
                                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                )
                                st.plotly_chart(fig, use_container_width=True)

            except GarminConnectAuthenticationError:
                st.error("Garmin authentication failed. Check email/password.")
            except Exception as e:
                st.error(f"Error while talking to Garmin: {e}")