import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    print(f"Key found: {api_key[:5]}...{api_key[-3:]}")
    try:
        genai.configure(api_key=api_key)
        print("🔄 Fetching available models...")
        models = genai.list_models()
        found = False
        print("\n-------- AVAILABLE MODELS --------")
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                print(f"{m.name}")
                found = True       
        if not found:
            print("No text generation models found. Check your API key permissions.")           
    except Exception as e:
        print(f"Error connecting to Google API: {e}")