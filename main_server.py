import sys
import os

# Ensure the server directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import threading
import time
from datetime import datetime
import requests
import re
import webbrowser  # For web navigation
import speech_recognition as sr  # For voice recognition
import pyttsx3  # For text-to-speech

# SDK imports
import firebase_admin
from firebase_admin import credentials, db

def extract_modules_from_question(question):
    modules = []
    q = question.lower()
    if any(kw in q for kw in ["eye", "gaze", "iris", "pupil", "eye tracking"]):
        modules.append("eye")
    if any(kw in q for kw in ["hand", "gesture", "hand gesture", "gesture control", "mouse control", "cursor"]):
        modules.append("hand")
    if any(kw in q for kw in ["sign", "sign language", "asl", "translator"]):
        modules.append("sign")
    return modules


# Import your controller classes with error handling
try:
    from hand_gestures import GestureController
    print("✅ GestureController imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import GestureController: {e}")
    GestureController = None

try:
    from eye_control import EyeController
    print("✅ EyeController imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import EyeController: {e}")
    EyeController = None

try:
    from sign_language_mouse import SignLanguageMouseController
    print("✅ SignLanguageMouseController imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import SignLanguageMouseController: {e}")
    SignLanguageMouseController = None

try:
    from chatbot_service import chatbot
    print("✅ ChatbotService imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import ChatbotService: {e}")
    chatbot = None


# --- AI/Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, 'qa_index.faiss')
QA_DATA_PATH = os.path.join(SCRIPT_DIR, 'qa_data.json')
TEMPLATE_FOLDER_PATH = os.path.join(SCRIPT_DIR, 'templates')
STATIC_FOLDER_PATH = os.path.join(SCRIPT_DIR, 'static')

print("--- Loading AI Models ---")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(FAISS_INDEX_PATH)
with open(QA_DATA_PATH, 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)
print("--- AI Models Loaded ---")

# --- FLASK APP SETUP ---
app = Flask(__name__, template_folder=TEMPLATE_FOLDER_PATH, static_folder=STATIC_FOLDER_PATH)
CORS(app)

# --- Firebase Initialization ---
# Use the absolute path directly as it is outside the script directory
FIREBASE_CRED_PATH = r"C:\Users\HP\Desktop\flutt\flutt (2)\flutt\flutt\gesturespace1-firebase-adminsdk-fbsvc-76dd72f9d2.json"
FIREBASE_DB_URL = 'https://gesturespace1-default-rtdb.firebaseio.com/'  # Replace with your Firebase DB URL

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DB_URL
    })

feedback_ref = db.reference('feedbacks')

def save_feedback(feedback_data):
    if 'submitted_at' not in feedback_data:
        feedback_data['submitted_at'] = datetime.utcnow().isoformat()
    print("Saving feedback:", feedback_data)
    feedback_ref.push(feedback_data)

# --- GLOBAL CONTROLLER MANAGEMENT ---
active_controller = None
lock = threading.Lock()

# --- Voice Assistant Setup ---
recognizer = sr.Recognizer()
is_voice_assistant_active = False
voice_thread = None
tts_lock = threading.Lock()  # Lock for TTS engine

# --- Website Dictionary for Navigation ---
WEBSITES = {
    "google": "www.google.com",
    "youtube": "www.youtube.com",
    "github": "www.github.com",
    "facebook": "www.facebook.com",
    "twitter": "www.twitter.com",
    "instagram": "www.instagram.com",
    "gmail": "www.gmail.com",
    "weather": "www.weather.com",
    "linkedin": "www.linkedin.com",
    "amazon": "www.amazon.com",
    "netflix": "www.netflix.com",
    "spotify": "www.spotify.com"
}

def stop_any_active_controller():
    global active_controller
    if active_controller:
        controller_name = type(active_controller).__name__
        print(f"🛑 Stopping {controller_name}...")
        try:
            active_controller.stop()
            time.sleep(0.5)
            active_controller = None
            print(f"✅ {controller_name} stopped successfully")
        except Exception as e:
            print(f"⚠️ Error while stopping controller: {e}")
            active_controller = None
        print("🔄 All controllers cleared")

def navigate_to_website(url):
    """Open a website in a new browser tab"""
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    try:
        print(f"🌐 Opening website: {url}")
        webbrowser.open_new_tab(url)
        return f"Opening {url}"
    except Exception as e:
        print(f"Error opening website: {e}")
        return f"Failed to open {url}: {str(e)}"

# 2. Replace your get_relevant_context function with this version:
def get_relevant_context(user_query, top_k=5):
    modules = extract_modules_from_question(user_query)
    if not modules:
        # If no module detected, fallback to top_k as before
        query_embedding = model.encode([user_query])
        D, I = index.search(np.array(query_embedding).astype('float32'), top_k)
        retrieved_texts = []
        for i, distance in zip(I[0], D[0]):
            if i < len(knowledge_base) and distance < 1.2:
                retrieved_texts.append(knowledge_base[i]['content'])
        return "\n\n".join(retrieved_texts)
    # If module(s) detected, only include context matching those modules
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding).astype('float32'), len(knowledge_base))
    retrieved_texts = []
    for i, distance in zip(I[0], D[0]):
        if i < len(knowledge_base) and distance < 1.3:
            content = knowledge_base[i]['content'].lower()
            if any(m in content for m in modules):
                retrieved_texts.append(knowledge_base[i]['content'])
        if len(retrieved_texts) >= top_k:
            break
    return "\n\n".join(retrieved_texts)




# 3. Replace your chat_with_novita function with this version:


def process_command(text):
    global active_controller
    text = text.lower().strip()
    print(f"Processing command: '{text}'")
    if active_controller is not None:
        print(f"🎯 Gesture module active: {type(active_controller).__name__}")
        if any(word in text for word in ["stop", "deactivate", "disable", "quit", "end", "close", "exit"]):
            print("🛑 Stop command detected - stopping gesture module")
            stop_any_active_controller()
            try:
                speak("Stopping all controllers")
            except:
                print("TTS: Stopping all controllers")
            return "All controllers stopped"
        else:
            print("🚫 Ignoring command - gesture module is active, only 'stop' commands allowed")
            return "Gesture module is active. Say 'stop' to deactivate."
    text = text.replace(" i ", " eye ").replace(" ai ", " eye ").replace(" aye ", " eye ")
    text = text.replace("i control", "eye control").replace("ai control", "eye control")
    text = text.replace("track i", "track eye").replace("tracking i", "tracking eye")
    text = text.replace("start i", "start eye").replace("start ai", "start eye ")
    text = text.replace("activate i", "activate eye").replace("enable i", "enable eye")
    text = text.replace("launch i", "launch eye").replace("run i", "run eye")
    text = text.replace("hand just", "hand gest").replace("hand chest", "hand gest")
    text = text.replace("gesture", "gesture").replace("jesture", "gesture")
    text = text.replace("sine", "sign").replace("sign language", "sign language")
    print(f"Normalized command: '{text}'")
    if any(word in text for word in ["open", "navigate", "go to", "browse"]):
        for site, url in WEBSITES.items():
            if site in text:
                response = navigate_to_website(url)
                try:
                    speak(f"Opening {site}")
                except:
                    print(f"TTS: Opening {site}")
                return response
        words = text.split()
        for word in words:
            if word not in ["open", "navigate", "to", "go", "browse", "the"]:
                if "." in str(word):
                    url = word
                    response = navigate_to_website(url)
                    try:
                        speak(f"Opening {url}")
                    except:
                        print(f"TTS: Opening {url}")
                    return response
    if "weather" in text:
        response = navigate_to_website("www.weather.com")
        try:
            speak("Opening weather information")
        except:
            print("TTS: Opening weather information")
        return "Opening weather information"
    start_words = ["start", "activate", "enable", "launch", "open", "begin", "run", "turn on", "switch on"]
    if any(word in text for word in start_words):
        print(f"🔍 Detected activation word in: '{text}'")
        hand_patterns = ["hand", "gesture", "mouse control", "cursor control", "hand control"]
        if any(pattern in text for pattern in hand_patterns) and "eye" not in text and "sign" not in text:
            print("🖐️ Starting hand gesture control")
            response = start_gesture_internal()
            try:
                speak("Starting hand gesture control")
            except:
                print("TTS: Starting hand gesture control")
            return "Hand gesture control activated"
        eye_patterns = ["eye control", "eye tracking", "eye track", "tracking eye", "track eye", 
                       "i control", "ai control", "gaze control", "eye mouse"]
        if any(pattern in text for pattern in eye_patterns):
            print("👁️ Starting eye control")
            response = start_eye_control_internal()
            try:
                speak("Starting eye tracking system")
            except:
                print("TTS: Starting eye tracking system")
            return "Eye tracking system activated"
        sign_patterns = ["sign", "sign language", "asl", "translator", "sign control"]
        if any(pattern in text for pattern in sign_patterns):
            print("🤟 Starting sign language control")
            response = start_sign_language_mouse_internal()
            try:
                speak("Starting sign language control")
            except:
                print("TTS: Starting sign language control")
            return "Sign language control activated"

    elif any(phrase in text for phrase in ["eye control", "eye tracking", "start eye", "activate eye", "enable eye"]):
        print("👁️ Fallback: Starting eye control")
        response = start_eye_control_internal()
        try:
            speak("Starting eye tracking system")
        except:
            print("TTS: Starting eye tracking system")
        return "Eye tracking system activated"
    print("💬 Processing as chat message")
    if chatbot:
        context = get_relevant_context(text)
        response = chatbot.get_response(text, context)
    else:
        response = "Chatbot service is not available."
    try:
        speak(response)
    except:
        print(f"TTS: {response}")
    return response

def start_gesture_internal():
    global active_controller, is_voice_assistant_active
    if GestureController is None:
        return "Error: Hand Gesture Controller not available"
    with lock:
        print("✅ Starting Hand Gesture Control internally.")
        try:
            stop_any_active_controller()
            is_voice_assistant_active = False
            print("🔇 Main voice assistant deactivated")
            active_controller = GestureController()
            thread = threading.Thread(target=active_controller.run, daemon=True)
            thread.start()
            print("✅ Hand Gesture Control thread started successfully.")
            return "Hand Gesture Control started."
        except Exception as e:
            print(f"❌ Error starting Hand Gesture Control: {e}")
            return f"Failed to start Hand Gesture Control: {str(e)}"

def start_eye_control_internal():
    global active_controller, is_voice_assistant_active
    if EyeController is None:
        return "Error: Eye Controller not available"
    with lock:
        print("✅ Starting Eye Control internally.")
        try:
            stop_any_active_controller()
            is_voice_assistant_active = False
            print("🔇 Main voice assistant deactivated")
            print("🔄 Creating EyeController instance...")
            active_controller = EyeController()
            print("🔄 Starting EyeController thread...")
            thread = threading.Thread(target=active_controller.run, daemon=True)
            thread.start()
            print("✅ Eye Control thread started successfully.")
            return "Eye Tracking System started."
        except Exception as e:
            print(f"❌ Error starting Eye Control: {e}")
            return f"Failed to start Eye Tracking System: {str(e)}"

def start_sign_language_mouse_internal():
    global active_controller, is_voice_assistant_active
    with lock:
        print("✅ Starting Sign Language Mouse Control.")
        try:
            stop_any_active_controller()
            is_voice_assistant_active = False
            print("🔇 Main voice assistant deactivated")
            from sign_language_mouse import start_simple_mouse_control
            thread = threading.Thread(target=start_simple_mouse_control, daemon=True)
            thread.start()
            return "✅ Sign Language Mouse Control started successfully!"
        except Exception as e:
            print(f"❌ Error starting sign language mouse: {e}")
            return f"Error: {str(e)}"

def start_sign_language_translator_internal():
    global active_controller, is_voice_assistant_active
    if SignLanguageMouseController is None:
        return "Error: Sign Language Translator Controller not available"
    with lock:
        print("✅ Starting Sign Language Translator internally.")
        try:
            stop_any_active_controller()
            is_voice_assistant_active = False
            print("🔇 Main voice assistant deactivated")
            active_controller = SignLanguageMouseController()
            active_controller.mode = "translator"
            thread = threading.Thread(target=active_controller.run, daemon=True)
            thread.start()
            return "✅ Sign Language Translator started with translation UI window."
        except Exception as e:
            print(f"Error starting Sign Language Translator: {e}")
            return f"Failed to start Sign Language Translator: {str(e)}"


def listen_for_commands():
    global is_voice_assistant_active
    try:
        speak("Voice assistant activated. How can I help you?")
    except Exception as e:
        print(f"Initial TTS error: {e}")
    while is_voice_assistant_active:
        try:
            with sr.Microphone() as source:
                print("🎤 Listening for commands...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                recognizer.energy_threshold = 200
                recognizer.pause_threshold = 0.8
                recognizer.dynamic_energy_threshold = True
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
            try:
                command = None
                for attempt in range(2):
                    try:
                        command = recognizer.recognize_google(audio, language='en-US')
                        break
                    except sr.UnknownValueError:
                        if attempt == 0:
                            print("🔄 Retrying speech recognition...")
                            continue
                        else:
                            raise
                if command:
                    print(f"✅ Recognized: '{command}'")
                try:
                    response = process_command(command)
                    try:
                        emit_response_to_frontend(command, response)
                    except Exception as e:
                        print(f"Error sending to frontend: {e}")
                except Exception as e:
                    print(f"Error processing command: {e}")
                    try:
                        speak("Sorry, I encountered an error processing your command.")
                    except:
                        print("TTS Error: Sorry, I encountered an error processing your command.")
            except sr.UnknownValueError:
                print("❌ Could not understand audio - please speak clearly")
                try:
                    speak("Sorry, I didn't understand. Please repeat your command.")
                except:
                    print("TTS: Sorry, I didn't understand. Please repeat your command.")
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                try:
                    speak("Speech recognition service is unavailable. Please try again.")
                except:
                    print("TTS: Speech recognition service is unavailable. Please try again.")
        except Exception as e:
            print(f"Error in voice recognition loop: {e}")
            time.sleep(1)
    print("Voice assistant stopped.")

def speak(text):
    try:
        with tts_lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        print(f"TTS Message: {text}")

def emit_response_to_frontend(user_message, bot_response):
    try:
        response = requests.post("http://127.0.0.1:5001/receive_voice_message", 
                      json={"user_message": user_message, "bot_response": bot_response},
                      timeout=2)
        if response.status_code == 200:
            print("✅ Response sent to frontend successfully")
        else:
            print(f"⚠️ Frontend response status: {response.status_code}")
    except requests.exceptions.Timeout:
        print("⚠️ Timeout sending to frontend")
    except requests.exceptions.ConnectionError:
        print("⚠️ Connection error sending to frontend")
    except Exception as e:
        print(f"⚠️ Error sending to frontend: {e}")

@app.route('/')
def home():
    with lock:
        stop_any_active_controller()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    user_question = request.json.get('question')
    if chatbot:
        context = get_relevant_context(user_question)
        answer = chatbot.get_response(user_question, context)
    else:
        answer = "Chatbot service is not available."
    return jsonify({"content": answer})

@app.route('/start_gesture', methods=['GET'])
def start_gesture():
    result = start_gesture_internal()
    return jsonify(status="Success", message=result)

@app.route('/start_eye_control', methods=['GET'])
def start_eye_control():
    try:
        print("🔧 Manual eye control start requested via HTTP")
        result = start_eye_control_internal()
        print(f"🔧 Eye control start result: {result}")
        return jsonify(status="Success", message=result)
    except Exception as e:
        print(f"❌ Error in start_eye_control route: {e}")
        return jsonify(status="Error", message=f"Failed to start Eye Control: {str(e)}"), 500

@app.route('/start_sign_language_mouse', methods=['GET'])
def start_sign_language_mouse():
    try:
        result = start_sign_language_mouse_internal()
        return jsonify(status="Success", message=result)
    except Exception as e:
        print(f"Error in start_sign_language_mouse route: {e}")
        return jsonify(status="Error", message=f"Failed to start Sign Language Mouse: {str(e)}"), 500

@app.route('/start_sign_language_translator', methods=['GET'])
def start_sign_language_translator():
    try:
        result = start_sign_language_translator_internal()
        return jsonify(status="Success", message=result)
    except Exception as e:
        print(f"Error in start_sign_language_translator route: {e}")
        return jsonify(status="Error", message=f"Failed to start Sign Language Translator: {str(e)}"), 500


@app.route('/sign_language_translator_state', methods=['GET'])
def sign_language_translator_state():
    """Return latest ASL translator state for the web UI.

    This is a lightweight polling endpoint; the translator runs server-side.
    """
    try:
        try:
            import sign_language_mouse
        except Exception as e:
            return jsonify(status="Error", message=f"sign_language_mouse import failed: {e}"), 500

        snapshot = {}
        try:
            snapshot = sign_language_mouse.get_translator_snapshot()
        except Exception as e:
            snapshot = {"running": False, "error": str(e)}

        return jsonify(status="Success", data=snapshot)
    except Exception as e:
        return jsonify(status="Error", message=str(e)), 500


@app.route('/stop_control', methods=['GET'])
def stop_control():
    with lock:
        stop_any_active_controller()
    return jsonify(status="Success", message="All controls stopped.")

@app.route('/start_voice_assistant', methods=['GET'])
def start_voice_assistant():
    global is_voice_assistant_active, voice_thread
    if not is_voice_assistant_active:
        is_voice_assistant_active = True
        voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
        voice_thread.start()
        return jsonify(status="Success", message="Voice assistant started.")
    else:
        return jsonify(status="Success", message="Voice assistant is already running.")

@app.route('/stop_voice_assistant', methods=['GET'])
def stop_voice_assistant():
    global is_voice_assistant_active
    is_voice_assistant_active = False
    return jsonify(status="Success", message="Voice assistant stopped.")

@app.route('/voice_command', methods=['POST'])
def voice_command():
    command = request.json.get('command', '')
    if not command:
        return jsonify({"response": "No command provided"})
    response = process_command(command)
    return jsonify({"response": response})

@app.route('/receive_voice_message', methods=['POST'])
def receive_voice_message():
    data = request.json
    return jsonify({"status": "received"})

@app.route('/check_voice_responses', methods=['GET'])
def check_voice_responses():
    return jsonify({"status": "ok", "voice_active": is_voice_assistant_active})

@app.route('/navigate', methods=['POST'])
def navigate():
    url = request.json.get('url', '')
    if not url:
        return jsonify({"response": "No URL provided"})
    try:
        result = navigate_to_website(url)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"response": f"Error opening website: {str(e)}"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get("message")
        if not user_message:
            return jsonify({"response": "Please provide a message."})
        if any(keyword in user_message.lower() for keyword in ["open", "navigate", "go to", "browse"]):
            for site, url in WEBSITES.items():
                if site in user_message.lower():
                    result = navigate_to_website(url)
                    return jsonify({"response": result})
        context = get_relevant_context(user_message)
        if chatbot:
            reply = chatbot.get_response(user_message, context)
        else:
            reply = "Chatbot service is not available."
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": "I'm here to help with GestureSpace questions! Please try asking again."})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()
        rating = int(data.get('rating', 0) or 0)
        if not email or not message:
            return jsonify({
                "success": False,
                "error": "Email and message are required fields"
            }), 400
        if not 0 <= rating <= 5:
            return jsonify({
                "success": False,
                "error": "Rating must be between 0 and 5"
            }), 400
        feedback_doc = {
            "name": name,
            "email": email,
            "message": message,
            "rating": rating,
            "submitted_at": datetime.utcnow().isoformat()
        }
        save_feedback(feedback_doc)
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!"
        })
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Server error processing feedback"
        }), 500

if __name__ == '__main__':
    print("--- Starting GESTURE SPACE Server on port 5001 ---")
    app.run(host='127.0.0.1', port=5001, debug=False)