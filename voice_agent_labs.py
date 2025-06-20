"""
voice_agent_labs.py
-------------------
Seamless, low-latency voice interaction loop for QuantAI Hospital Assistant.

- Continuously listens for user voice input (microphone)
- Transcribes speech to text (using SpeechRecognition + Whisper or Google STT)
- Passes text to agent.py for context-aware response
- Converts agent response to speech using ElevenLabs (Rachel voice)
- Plays response aloud, with parallelism and low latency
- Robust error handling and user prompts

Example usage:
    python voice_agent_labs.py
"""
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
import agent  # Assumes agent.py is in the same directory and exposes an async main/agent class
import asyncio
import tempfile
from pathlib import Path
import io
import uuid
import wave
from typing import Optional, Dict, Any, Tuple
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ElevenLabs config
RACHEL_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
FASTEST_MODEL = "eleven_turbo_v2_5"
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVEN_API_KEY:
    raise EnvironmentError("ELEVENLABS_API_KEY not set in environment.")
# Initialize ElevenLabs client
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Thread pool for TTS
_tts_executor = ThreadPoolExecutor(max_workers=4)

# Recognizer for STT
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Async event loop for agent
_agent_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_agent_loop)

# Initialize the agent (singleton)
_agent_instance = None
def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = agent.QuantAIHospitalAgent()
    return _agent_instance

async def get_agent_response(text: str) -> str:
    agent_obj = get_agent()
    return await agent_obj.generate_response(text)

def tts_speak(text: str):
    """Generate and play TTS audio using ElevenLabs (Rachel)."""
    def _tts_job():
        try:
            audio = tts_client.generate(
                text=text,
                voice=RACHEL_VOICE_ID,
                model=FASTEST_MODEL
            )
            tts_client.play(audio)
            logger.info("Audio played successfully.")
        except Exception as e:
            logger.error(f"TTS generation/playback failed: {e}")
    _tts_executor.submit(_tts_job)

def listen_and_respond():
    print("\nKia ora! Voice assistant is ready. Speak into your microphone.")
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("\nListening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
            try:
                print("Transcribing...")
                # Try Whisper first, fallback to Google if not available
                try:
                    text = recognizer.recognize_whisper(audio, language="en")
                except Exception:
                    text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that. Please try again.")
                continue
            except sr.RequestError as e:
                print(f"STT error: {e}")
                continue
            # Get agent response (async, but block for result)
            print("Agent is thinking...")
            try:
                response = _agent_loop.run_until_complete(get_agent_response(text))
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Agent error: {e}")
                continue
            # Speak response (in parallel)
            tts_speak(response)
        except KeyboardInterrupt:
            print("\nExiting voice assistant. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("An error occurred. Please try again.")

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

class SpeechToText:
    """Speech-to-text conversion using various engines."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.default_language = "en"
        
    def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Tuple[str, float]:
        """Transcribe audio data to text using Whisper or Google STT as fallback.
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Ensure we have valid WAV data
            audio_data = self._validate_audio(audio_data)
            if not audio_data:
                raise ValueError("Invalid audio data")
                
            # Convert to AudioData for recognizer
            audio = sr.AudioData(audio_data, 16000, 2)
            
            # Try Whisper first
            try:
                text = self.recognizer.recognize_whisper(audio, language=language)
                return text, 0.9  # Whisper doesn't provide confidence scores
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}. Falling back to Google.")
                
            # Fallback to Google
            text = self.recognizer.recognize_google(audio, language=language)
            return text, 0.8  # Google doesn't provide confidence in this API
            
        except sr.UnknownValueError:
            logger.warning("Speech not recognized")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"STT service error: {e}")
            return "", 0.0
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0
    
    def _validate_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Validate and convert audio to compatible format if needed."""
        try:
            # Try to open as WAV
            with io.BytesIO(audio_data) as f:
                with wave.open(f, 'rb') as wav:
                    # Just checking if it's valid
                    pass
                return audio_data
        except Exception:
            # Try to convert with pydub
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                wav_io = io.BytesIO()
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set sample rate
                audio.export(wav_io, format='wav')
                return wav_io.getvalue()
            except Exception as e:
                logger.error(f"Audio conversion failed: {e}")
                return None

class TextToSpeech:
    """Text-to-speech conversion using ElevenLabs."""
    
    def __init__(self):
        self.eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.eleven_api_key:
            logger.warning("ELEVENLABS_API_KEY not set. TTS functionality will be limited.")
        self.client = ElevenLabs(api_key=self.eleven_api_key) if self.eleven_api_key else None
        self.default_voice = RACHEL_VOICE_ID
        self.default_model = FASTEST_MODEL
        self.temp_dir = TEMP_DIR
        
    def generate_speech(self, text: str, voice_id: str = None, model: str = None) -> Optional[bytes]:
        """Generate speech audio from text.
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (default: Rachel)
            model: ElevenLabs model name
            
        Returns:
            Audio data as bytes or None if generation failed
        """
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return None
            
        try:
            voice = voice_id or self.default_voice
            model_name = model or self.default_model
            
            audio = self.client.generate(
                text=text,
                voice=voice,
                model=model_name
            )
            return audio
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def save_audio_file(self, audio_data: bytes) -> Optional[str]:
        """Save audio data to a temporary file.
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Filename of saved audio or None if save failed
        """
        if not audio_data:
            return None
            
        try:
            filename = f"{uuid.uuid4()}.mp3"
            filepath = self.temp_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(audio_data)
                
            return filename
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None
    
    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data.
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            True if playback succeeded, False otherwise
        """
        if not audio_data:
            return False
            
        try:
            if self.client:
                self.client.play(audio_data)
                return True
            return False
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False

class VoiceAgent:
    """Voice agent for QuantAI Hospital Assistant."""
    
    def __init__(self):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.current_language = "en"
        self._initialize_agent()
        
    def _initialize_agent(self):
        """Initialize the text agent for responses."""
        try:
            self.agent = agent.QuantAIHospitalAgent()
            logger.info("Voice agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice agent: {e}")
            self.agent = None
    
    async def process_audio(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """Process audio input and generate a response.
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (default: agent's current language)
            
        Returns:
            Dictionary with user_text, response_text, and audio_url
        """
        result = {
            "user_text": "",
            "response_text": "",
            "audio_url": None,
            "error": None
        }
        
        try:
            # Set language
            lang = language or self.current_language
            
            # Transcribe audio
            user_text, confidence = self.stt.transcribe_audio(audio_data, lang)
            if not user_text:
                result["error"] = "Could not transcribe audio"
                return result
                
            result["user_text"] = user_text
            
            # Get response from agent
            if self.agent:
                response_text = await get_agent_response(user_text)
                result["response_text"] = response_text
                
                # Generate speech
                audio = self.tts.generate_speech(response_text)
                if audio:
                    filename = self.tts.save_audio_file(audio)
                    if filename:
                        result["audio_url"] = f"/audio/{filename}"
                
            else:
                result["error"] = "Agent not initialized"
                
            return result
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            result["error"] = f"Processing error: {str(e)}"
            return result
    
    def set_language(self, language: str):
        """Set the agent's language.
        
        Args:
            language: Language code (e.g., 'en', 'es')
        """
        self.current_language = language
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        # Basic set of supported languages
        return ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar", "hi"]

if __name__ == "__main__":
    listen_and_respond() 