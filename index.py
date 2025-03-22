import google.generativeai as genai
import os
from IPython.display import Markdown

genai.configure(api_key="AIzaSyBib0RPcR6NQpCPuiacab3WcZrUn2IV75E")

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("apa pengertian iot?")

print(response.text)
