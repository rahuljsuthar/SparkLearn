import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
print("Testing API Key:", api_key)

genai.configure(api_key=api_key)

try:
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Error listing models:", e)

try:
    print("\nTrying generate content with gemini-2.5-flash-lite...")
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content("Say hello!")
    print("Response text:", response.text)
except Exception as e:
    print("Error with gemini-2.5-flash-lite:", e)
