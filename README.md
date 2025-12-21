# Garmin program description
analyze Garmin activities and weight

# Install Command
pip install dash plotly pandas openpyxl streamlit
pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
#pip install selenium webdriver-manager requests gpxpy folium pandas python-dateutil tqdm

# Run Comman
python App.py
streamlit run App.py

# Create requirements.txt to hold the library
pip freeze > requirements.txt  

# Commit
git add App.py Weight_20251221.xlsx Activities_Run_20251202.csv requirements.txt README.md
git commit -m "update"  
git push  

