import google.generativeai as genai
import os
import subprocess
from gtts import gTTS
import markdown


genai.configure(api_key="AIzaSyBib0RPcR6NQpCPuiacab3WcZrUn2IV75E")

model = genai.GenerativeModel("gemini-2.0-flash",
                            system_instruction="""kamu adalah seorang virtual assistance
                            yang mampu menjawab pertanyaan user secara singkat dan jelas""")

response = model.generate_content("apa fungsi dari laptop")

hasil = response.text
language = "id"

file = gTTS(text=hasil, lang=language)

file.save("test.mp3")
print("file tersimpan")

subprocess.run(["start", "result.mp3"], shell=True)