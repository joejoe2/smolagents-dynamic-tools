import os
from gradio_ui import MyGradioUI
from dotenv import load_dotenv

# setting
load_dotenv(".env")
api_key = os.getenv("openai_api_key", "xxxxx")
api_base = os.getenv("openai_api_base", "https://openrouter.ai/api/v1")
model = os.getenv("model_id", "google/gemini-2.0-flash-exp:free")
authorized_imports = os.getenv(
    "authorized_imports", "matplotlib,matplotlib.pyplot,PIL,io,numpy,base64"
).split(sep=",")
session_ttl = os.getenv("session_ttl", None)
session_capacity = int(os.getenv("session_capacity", 10000))

gradio = MyGradioUI(
    api_base=api_base,
    api_key=api_key,
    model=model,
    authorized_imports=authorized_imports,
    session_ttl=int(session_ttl) if session_ttl else None,
    session_capacity=session_capacity,
)
if __name__ == "__main__":
    gradio.launch()
