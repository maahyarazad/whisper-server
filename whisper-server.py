from flask import Flask, request, jsonify
import whisper
import requests
import os
from flask_cors import CORS
import logging
import uuid
from gtts import gTTS
from flask import send_file
import io

logging.basicConfig(level=logging.DEBUG)



app = Flask(__name__)
CORS(app)  # Allow CORS for frontend requests

# Load the Whisper model once
whisper_model = whisper.load_model("base")

API_KEY = os.getenv('TRANSCRIBE_API_KEY', 'your-secret-key')


@app.before_request
def require_api_key():
    if request.endpoint == 'transcribe_and_send':
        api_key = request.headers.get('X-API-KEY')
        if api_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        
@app.route('/api/transcribe', methods=['POST'])
def transcribe_and_send():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_path = os.path.join("temp_audio", audio_file.filename)
    os.makedirs("temp_audio", exist_ok=True)
    audio_file.save(temp_path)

    # Transcribe
    result = whisper_model.transcribe(temp_path, language="en")
    transcription = result['text']

    # Send to Ollama
    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": transcription,
                "stream": False
            }
        )
        ollama_response.raise_for_status()
        ollama_data = ollama_response.json()
        response_text = ollama_data.get("response", "No response key found")
        logging.debug(ollama_data)
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp3"

        # Create and save TTS audio
        tts = gTTS(ollama_response.json()["response"])
        #tts.save(filename)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

    except Exception as e:

        response_text = f"Error contacting Ollama: {str(e)}"

    return send_file(
        mp3_fp,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name="response.mp3"
    )
    #return jsonify({
    #    "transcription": transcription,
    #    "ollama_response": ollama_response.json()["response"]
    #})

if __name__ == '__main__':
    app.run(debug=True, port=6000)
