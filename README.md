# Garmin program description
analyze Garmin activities and weight

# Install Command
pip install dash plotly pandas openpyxl streamlit
pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
#pip install selenium webdriver-manager requests gpxpy folium pandas python-dateutil tqdm

python App.py
streamlit run App.py
python RaceResult.py
streamlit run RaceResult.py

# Create requirements.txt to hold the library
pip freeze > requirements.txt  

# Commit
git add App.py Weight_20260111.xlsx requirements.txt README.md
git commit -m "v20260401: enhance login failure error handling"  
git push  

