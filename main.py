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

gradio = MyGradioUI(
    api_base=api_base,
    api_key=api_key,
    model=model,
    authorized_imports=authorized_imports,
)
if __name__ == "__main__":
    gradio.launch()
