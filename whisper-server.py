from flask import Flask, request, jsonify, session
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
import subprocess
import json
import time
from collections import defaultdict
import copy
import threading


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

user_histories = defaultdict(list) 

# Load the Whisper model once
# whisper_model = whisper.load_model("base")

API_KEY = os.getenv('TRANSCRIBE_API_KEY', 'your-secret-key')


initial_prompt = "limit the context of this coversation to software development domain. " \
        "customer service of a software dev company, answer the " \
        "question of the user if it relates to our services. here is the list of our services: " \
        "Internet of Things Development: builds custom IoT solutions. We can consult on building IoT infrastructure or develop high-quality IoT software for enterprises and startups, focused on connected devices that you can control and manage via web and mobile applications." \
        "Software Product Development: We create scalable and high-performing software products tailored to your business needs. From MVPs for startups to full-scale enterprise applications, our solutions are designed to drive growth, enhance efficiency, and meet market demands." \
        "Web Development: Our web development services deliver fast, secure, and user-friendly websites and applications. We develop web projects that enable businesses to meet user expectations, drive growth, and stay competitive in an ever-changing marketplace." \
        "Mobile App Development: We offer custom mobile app development services, specializing in iOS and Android platforms, creating user-friendly native, hybrid, and progressive web apps that cater to business needs and engage users." \
        "IT Staff Augmentation: Expand your team with our skilled teams of software developers, business analysts, UX/UI designers, or QA talents. We ensure smooth onboarding, expertise of top professionals, on-demand scaling, and seamless integration with your in-house team for short-term or long-term periods." \
        "After one or two prompts, ask the user whether they would like to book a meeting with us." \
        "If someone asks you to book a meeting for them, make sure not to talk about our services and" \
        "Get the date and verify that it is not a Saturday or Sunday. and accepts on or at perposition for dates"\
        "Ensure the request includes the person's name, phone number"\
        "Once you have all the required information, just book the meeting and don't ask about anything."


initial_assistant_reply = ""

initial_chat_history = []

def send_to_ollama_chat(prompt , user_id, model="llama3"):

    message = None

    if user_id == None: 
        message= [{"role": "user", "content": prompt}]
    else: 
        message = user_histories[user_id]

    try:
        with requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": message ,
                "stream": True
            },
            stream=True
        ) as response:
            response.raise_for_status()

            assistant_reply = ""

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        assistant_reply += data.get("message", {}).get("content", "")
                    except Exception as e:
                        print(f"Decoding error: {e}")

        # # Append assistant response
        # chat_history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return "Error communicating with model"


def send_to_ollama_generate(prompt):
    response_text = ""

    with requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    ) as response:
        response.raise_for_status()
        
        # Stream and accumulate the response
        for line in response.iter_lines():
            if line:
                try:
                    # Decode JSON line-by-line (each line is a partial chunk of the response)
                    chunk = line.decode("utf-8")
                    json_data = json.loads(chunk)
                    response_text += json_data.get("response", "")
                except Exception as e:
                    logging.warning(f"Failed to decode chunk: {chunk} | Error: {e}")

    logging.debug(response_text)
    return response_text

def send_webhook(payload):
    try:
        requests.post('http://localhost:5000/api/meeting-request', json=payload, timeout=5)
    except Exception as e:
        print("Webhook failed:", e)

        
@app.route('/api/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_and_send():
    try:
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200  # Preflight success

        # Check API key only for POST
        if request.headers.get('X-API-KEY') != 'your-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400


        ffmpeg_start = time.perf_counter()

        audio_file = request.files['audio']
        input_path = os.path.join("temp_audio", audio_file.filename)
        os.makedirs("temp_audio", exist_ok=True)
        audio_file.save(input_path)

        whisper_cpp_path = os.path.join("whisper-cpp", "whisper.cpp", "build", "bin", "whisper-cli")
        model_path = os.path.join("whisper-cpp", "whisper.cpp", "models", "ggml-base.en.bin")

        wav_path = "temp_audio/recording.wav"

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, wav_path
        ], check=True)

        ffmpeg_end = time.perf_counter()  
        logging.debug(f"FFmpeg Process Elapsed time: {ffmpeg_end - ffmpeg_start:.4f} seconds")


        whisper_start = time.perf_counter()  
        # Run whisper.cpp
        whisper_process = subprocess.run(
            [
                whisper_cpp_path,
                "-m", model_path,
                "-f", wav_path,
                "-l", "en",
                "-otxt",  # Write to text file
                "-nt"     # (optional) omit timestamps for cleaner text output
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        transcription = None

        if whisper_process.returncode == 0:
            transcription = whisper_process.stdout.decode('utf-8')
        else:
            return jsonify({'error': f'{whisper_process.stderr.strip()}'}), 400


        whisper_end = time.perf_counter()  
        logging.debug(f"Whisper Process Elapsed time: {whisper_end - whisper_start:.4f} seconds")
        logging.debug(transcription)
        # # Send to Ollama
    

        
        logging.debug(transcription)
        ollama_start = time.perf_counter()  


        # "{\n\"date\": \"\",\n\"name\": \"\",\n\"phoneNumber\": \"\",\n\"requestTopic\" : \"\"\n\n}"\



        # full_prompt = f"{custom_data}\n\nAnswer this question based on the above data:\n{transcription}"


        # process = subprocess.run(
        #     ["ollama", "run", "llama3.2", transcription],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True  # so stdout/stderr are strings instead of bytes
        # )

        # response_text = None

        # if process.returncode == 0:
        #     response_text = process.stdout.strip()
        # else:
        #     response_text = f"Error: {process.stderr.strip()}"

        # logging.debug(response_text)
        # ollama_end = time.perf_counter()  
        # logging.debug(f"Ollama Process Elapsed time: {ollama_end - ollama_start:.4f} seconds")

        ollama_api_start = time.perf_counter()  
        user_id = request.headers.get("X-Session-ID")

        if user_histories.__len__() == 0:
            user_histories[user_id] = copy.deepcopy(initial_chat_history)
                  
        user_histories[user_id].append({"role": "user", "content": transcription})

        # Call Ollama
        assistant_response = send_to_ollama_chat( None , user_id)

        # Append assistant response
        user_histories[user_id].append({"role": "assistant", "content": assistant_response})
        


        logging.debug(assistant_response)
        ollama_api_end = time.perf_counter()  
        logging.debug(f"Ollama Process Elapsed time: {ollama_api_end - ollama_api_start:.4f} seconds")

        if "I've booked a meeting" in assistant_response:
            threading.Thread(target=send_webhook, args=({"message": assistant_response,})).start()


        return jsonify({
           "ollama_response": assistant_response
        })

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
    
    initial_chat_history.append({"role": "user", "content": initial_prompt})
    
    assistant_reply = send_to_ollama_chat(initial_prompt, None)
    
    initial_chat_history.append({"role": "assistant", "content": assistant_reply})

    app.run(host='0.0.0.0', debug=False, port=8081, threaded=True)

