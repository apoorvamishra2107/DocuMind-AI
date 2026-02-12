import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(prompt):
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text
