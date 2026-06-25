# Streamlit Cloud entry point
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
exec(open(os.path.join(os.path.dirname(__file__), "src", "app.py")).read())
