# Garmin program description
analyze Garmin activities and weight

# Install Command
pip install dash plotly pandas openpyxl streamlit

# Run Comman
python App.py
streamlit run App.py

# Create requirements.txt to hold the library
pip freeze > requirements.txt  

# Commit
git add App.py Weight_20250928.xlsx Activities_Run.csv requirements.txt README.md
git commit -m "Add Python script, Excel data, and requirements"  
git push  

