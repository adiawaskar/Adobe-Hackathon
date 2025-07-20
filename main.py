import subprocess
import os
from pathlib import Path


# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INPUT_DIR = PROJECT_ROOT / "input"

# Change this to test another PDF
PDF_NAME = "sample5.pdf"
PDF_PATH = INPUT_DIR / PDF_NAME

# Ensure the PDF exists
if not PDF_PATH.exists():
    raise FileNotFoundError(f"❌ PDF not found at path: {PDF_PATH}")

print("\n🔍 Step 1: Ingesting PDF...")
subprocess.run(["python", str(SCRIPTS_DIR / "01_pdf_ingestion.py"), str(PDF_PATH)], check=True)

print("\n🧱 Step 2: Extracting Text Blocks...")
subprocess.run(["python", str(SCRIPTS_DIR / "02_feature_engineering.py")], check=True)

print("\n🔗 Step 3: Merging Headings...")
subprocess.run(["python", str(SCRIPTS_DIR / "03_heading_classification.py")], check=True)

os.environ["PDF_NAME"] = PDF_NAME 

print("\n🤩 Step 4: Creating Structured Outline...")
subprocess.run(["python", str(SCRIPTS_DIR / "04_outline_structuring.py")], check=True)

print("\n🚀 All steps completed. Check the output folder!")
