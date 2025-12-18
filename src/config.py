import os

BASE_DIR = "data"
INFO_DIR = os.path.join(BASE_DIR,"info")
PG_DIR = os.path.join(BASE_DIR,"pg")
UG_DIR = os.path.join(BASE_DIR,"ug")



LLM_MODEL = "gemini-2.5-flash"  
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
