import requests
import re
import os

class ChatbotService:
    def __init__(self):
        self.api_key = "sk_X8KX9wyoRWA4zfA3Iomz1zGZdRq3naKSYt1EPUYYvck" # Default/Existing Key
        self.provider = "novita" # Default provider, can be switched to 'gemini' or 'openai' later
        
        # Novita config
        self.novita_url = "https://api.novita.ai/v3/openai/chat/completions"
        self.novita_model = "qwen/qwen-2.5-72b-instruct" # Upgraded model if possible, or keep existing

    def _extract_modules(self, question):
        """Analyze question to determine relevant modules."""
        modules = []
        q = question.lower()
        if any(kw in q for kw in ["eye", "gaze", "iris", "pupil", "eye tracking"]):
            modules.append("eye")
        if any(kw in q for kw in ["hand", "gesture", "hand gesture", "gesture control", "mouse control", "cursor"]):
            modules.append("hand")
        if any(kw in q for kw in ["sign", "sign language", "asl", "translator"]):
            modules.append("sign")
        return modules

    def _get_system_prompt(self, modules):
        """Generate a system prompt based on detected modules."""
        base_prompt = (
            "You are an AI assistant for GestureSpace, a hands-free desktop control app.\n"
            "Your goal is to provide helpful, concise, and accurate answers about the app's features.\n"
            "Keep answers under 4 sentences. Be friendly but direct.\n"
        )

        if modules:
            module_names = {
                "eye": "Eye Tracking System (controls mouse with gaze)",
                "hand": "Hand Gesture Control (controls mouse with hand movements)",
                "sign": "Sign Language Mouse & Translator (ASL gestures)",
            }
            focused_features = ", ".join([module_names[m] for m in modules])
            base_prompt += f"The user is asking about: {focused_features}. Focus your answer on these features."
        else:
            base_prompt += "Answer general questions about GestureSpace features (Eye Tracking, Hand Gestures, Sign Language Control)."

        return base_prompt

    def _clean_response(self, text):
        """Clean up the response text."""
        if not text:
            return "I'm here to help with GestureSpace."
        
        # Remove thinking/reasoning traces often found in some models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*.*?\*\*', '', text, flags=re.DOTALL) # Remove bolding for TTS clarity
        
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            # Filter out meta-speech
            if any(p in line.lower() for p in ["okay", "let me check", "here is the answer", "user asks"]):
                continue
            clean_lines.append(line)
            
        return " ".join(clean_lines[:4]) # Limit to 4 sentences/lines

    def get_response(self, user_message, context=""):
        """Get a response from the chatbot."""
        try:
            modules = self._extract_modules(user_message)
            system_prompt = self._get_system_prompt(modules)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_message}"}
            ]

            if self.provider == "novita":
                return self._call_novita(messages)
            
            return "Chatbot provider not configured correctly."

        except Exception as e:
            print(f"Chatbot Error: {e}")
            return "I'm having trouble connecting to my brain right now. Please try again."

    def _call_novita(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.novita_model,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.5,
        }
        
        try:
            response = requests.post(self.novita_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return self._clean_response(content)
            else:
                print(f"Novita API Error: {response.text}")
                return "I couldn't reach the AI server."
        except Exception as e:
            print(f"Novita Request Failed: {e}")
            return "Connection to AI server failed."

# Singleton instance
chatbot = ChatbotService()
