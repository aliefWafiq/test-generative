import google.generativeai as genai
import os
import subprocess
from gtts import gTTS
import markdown
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["POST"])
def receive():
    genai.configure(api_key="AIzaSyBib0RPcR6NQpCPuiacab3WcZrUn2IV75E")

    model = genai.GenerativeModel("gemini-2.0-flash",
                            system_instruction="""nama kamu adalah zero kamu seorang virtual assistance
                            yang mampu menjawab pertanyaan user  dalam bahasa indonesia secara singkat dan jelas tanpa menggunakan simbol dan hanya huruf atau angka saja""")

    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "invalid"}), 400

    keyword = data["text"]
    print(keyword)

    response = model.generate_content(keyword)

    hasil = response.text
    language = "id"

    file = gTTS(text=hasil, lang=language)

    file.save("test.mp3")
    print("file tersimpan")

    subprocess.run(["start", "test.mp3"], shell=True)
    return jsonify({"status": "sukses", "received": keyword}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)