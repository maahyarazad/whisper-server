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
from TTS.api import TTS
import soundfile as sf 
import torch
logging.basicConfig(level=logging.DEBUG)



app = Flask(__name__)
CORS(app)  # Allow CORS for frontend requests

# Load the Whisper model once
whisper_model = whisper.load_model("tiny.en")

API_KEY = os.getenv('TRANSCRIBE_API_KEY', 'your-secret-key')


@app.before_request
def require_api_key():
    if request.endpoint == 'transcribe_and_send':
        api_key = request.headers.get('X-API-KEY')
        if api_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        
@app.route('/api/transcribe', methods=['POST'])
def transcribe_and_send():
    try:
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


        # 1. Instantiate without “gpu” flag:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

        # 2. Move it onto CUDA (if you have a GPU), or leave it on CPU:
        #    – for GPU:
        # If you have a GPU, move it there:
        if torch.cuda.is_available():
            tts.to("cuda")
        else:
            tts.to("cpu")

        output_path = "output.wav"

        # If file exists, remove it so tts_to_file can overwrite
        if os.path.exists(output_path):
            os.remove(output_path)
            
        tts.tts_to_file(text=ollama_response.json()["response"], file_path=output_path)


        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="response.wav",
        )

    except Exception as e:
        logging.exception("Error in /api/transcribe")
        response_text = f"Error: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True, port=6000)
