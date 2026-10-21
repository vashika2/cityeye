"""Streamlit Cloud entrypoint redirecting to dashboard/app.py."""
import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

app_path = ROOT_DIR / "dashboard" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
