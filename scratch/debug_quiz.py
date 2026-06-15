import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
print("Loaded API key:", GEMINI_API_KEY[:10] + "..." if GEMINI_API_KEY else "None")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

topic_name = "Mixed CS"
difficulty = "mixed"
count = 10

prompt = (f"Generate {count} MCQ on '{topic_name}', difficulty:{difficulty}.\n"
          f"JSON array only:\n"
          f'[{{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"..."}}]')

print("--- Generating with max_tokens=900 ---")
try:
    resp = model.generate_content(
        contents=prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=900, temperature=0.7))
    print("Length of raw response:", len(resp.text))
    print("Raw text:")
    print(resp.text)
except Exception as e:
    print("Error with max_tokens=900:", e)

print("--- Generating with max_tokens=3000 ---")
try:
    resp = model.generate_content(
        contents=prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=3000, temperature=0.7))
    print("Length of raw response:", len(resp.text))
    print("Raw text:")
    print(resp.text)
except Exception as e:
    print("Error with max_tokens=3000:", e)
