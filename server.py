"""
QuantAI Hospital API Server
This module provides a FastAPI server that integrates both text and voice processing capabilities
from the QuantAI Hospital AI Assistant system.
"""

import os
import logging
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import tempfile
import uuid
from datetime import datetime
import speech_recognition as sr
import io
import wave
import time
import hashlib
import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from deep_translator.exceptions import LanguageNotSupportedException
from cachetools import TTLCache
from contextlib import asynccontextmanager

# Import our agent implementations
from agent import QuantAIHospitalAgent
from voice_agent_labs import VoiceAgent, TextToSpeech, SpeechToText

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Add required classes to safe globals for PyTorch 2.6+
import torch.serialization
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Australian Police API Server")
    cleanup_old_files(TEMP_DIR)
    try:
        logger.info("Initializing Australian Police API Server...")
        text_agent = QuantAIHospitalAgent()
        voice_agent = VoiceAgent()
        # Set default language without prompting
        text_agent.user_language = "en"  # Default to English
        voice_agent.current_language = "en"  # Default to English
        logger.info("✓ Text agent initialized")
        logger.info("✓ Voice agent initialized")
        logger.info("✓ Application Startup Complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise RuntimeError(f"Application initialization failed: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Australian Police API Server")
    cleanup_old_files(TEMP_DIR)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Australian Police API",
    description="API server for Australian Police's text and voice processing capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our agents
logger.info("Initializing QuantAI Hospital API Server...")
text_agent = QuantAIHospitalAgent()
voice_agent = VoiceAgent()
# Set default language without prompting
text_agent.user_language = "en"  # Default to English
voice_agent.current_language = "en"  # Default to English
logger.info("✓ Text agent initialized")
logger.info("✓ Voice agent initialized")
logger.info("✓ Application Startup Complete")

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Create batch translation cache
batch_translation_cache = TTLCache(maxsize=500, ttl=3600)  # Cache for 1 hour

class TextQuery(BaseModel):
    """Model for text query requests."""
    text: str
    language: str = "english"  # Required with default
    auto_detect_language: Optional[bool] = Field(default=False, description="Automatically detect the input language")
    translate_input: Optional[bool] = Field(default=False, description="Translate the input to English before processing")

class LanguageDetectionRequest(BaseModel):
    """Model for language detection requests."""
    text: str

class LanguageDetectionResponse(BaseModel):
    """Model for language detection responses."""
    detected_language: str
    confidence: Optional[float] = None
    success: bool
    error: Optional[str] = None

class BatchTranslationRequest(BaseModel):
    """Model for batch translation requests."""
    texts: list[str]
    target_language: str
    source_language: str = "english"  # Changed from Optional to required with default

class BatchTranslationResponse(BaseModel):
    """Model for batch translation responses."""
    success: bool
    translations: list[str]
    target_language: str
    failed_indices: Optional[list[int]] = None
    error: Optional[str] = None

class TextResponse(BaseModel):
    """Model for text query responses."""
    success: bool
    response: str
    language: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None

class VoiceResponse(BaseModel):
    """Model for voice query responses."""
    user_text: str
    response_text: str
    audio_url: Optional[str] = None
    error: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    """Model for text-to-speech requests."""
    text: str
    language: Optional[str] = "english"

class TextToSpeechResponse(BaseModel):
    """Model for text-to-speech responses."""
    success: bool
    audio_url: str
    text: str
    error: Optional[str] = None

def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    """Clean up old temporary files."""
    current_time = datetime.now().timestamp()
    for file in directory.glob("*"):
        if current_time - file.stat().st_mtime > (max_age_hours * 3600):
            try:
                file.unlink()
                logger.info(f"Cleaned up old file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {e}")

def validate_and_convert_audio(audio_data: bytes) -> Optional[bytes]:
    """Try to open the audio file as WAV, or convert to WAV if needed."""
    try:
        # Try to open as WAV
        with io.BytesIO(audio_data) as f:
            with wave.open(f, 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                logger.info(f"WAV file validated: channels={channels}, sample_width={sample_width}, framerate={framerate}")
                return audio_data
    except Exception as e:
        logger.warning(f"Audio file is not a valid WAV: {e}. Attempting conversion with pydub.")
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            logger.info("Audio file converted to WAV using pydub.")
            return wav_io.getvalue()
        except Exception as e2:
            logger.error(f"Audio file could not be processed: {e2}")
            return None

def get_xtts_language_code(language: str) -> str:
    """Map language name or code to XTTS v2 code."""
    language = (language or "english").strip().lower()
    mapping = {
        'english': 'en', 'en': 'en',
        'spanish': 'es', 'es': 'es',
        'french': 'fr', 'fr': 'fr',
        'german': 'de', 'de': 'de',
        'hindi': 'hi', 'hi': 'hi',
        'chinese': 'zh-cn', 'zh': 'zh-cn', 'zh-cn': 'zh-cn',
        'japanese': 'ja', 'ja': 'ja',
        'italian': 'it', 'it': 'it',
        'russian': 'ru', 'ru': 'ru',
        'arabic': 'ar', 'ar': 'ar',
        'portuguese': 'pt', 'pt': 'pt',
        'korean': 'ko', 'ko': 'ko',
        'polish': 'pl', 'pl': 'pl',
        'turkish': 'tr', 'tr': 'tr',
        'dutch': 'nl', 'nl': 'nl',
        'czech': 'cs', 'cs': 'cs',
        'hungarian': 'hu', 'hu': 'hu',
    }
    return mapping.get(language, 'en')

def map_code_to_deep_translator_language(language_code: str) -> str:
    """Map short language codes or names to DeepTranslator's expected language names."""
    code_map = {
        'en': 'english', 'english': 'english',
        'es': 'spanish', 'spanish': 'spanish',
        'fr': 'french', 'french': 'french',
        'de': 'german', 'german': 'german',
        'it': 'italian', 'italian': 'italian',
        'pt': 'portuguese', 'portuguese': 'portuguese',
        'zh': 'chinese', 'zh-cn': 'chinese', 'chinese': 'chinese',
        'ja': 'japanese', 'japanese': 'japanese',
        'ko': 'korean', 'korean': 'korean',
        'ar': 'arabic', 'arabic': 'arabic',
        'ru': 'russian', 'russian': 'russian',
        'hi': 'hindi', 'hindi': 'hindi',
        'pl': 'polish', 'polish': 'polish',
        'tr': 'turkish', 'turkish': 'turkish',
        'nl': 'dutch', 'dutch': 'dutch',
        'cs': 'czech', 'czech': 'czech',
        'hu': 'hungarian', 'hungarian': 'hungarian',
    }
    return code_map.get(language_code.lower().strip(), 'english')

@app.get("/languages-text")
async def get_text_languages():
    """
    Get list of available languages for text translation.
    """
    try:
        # Map of language codes to display names
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'ru': 'Russian',
            'hi': 'Hindi',
            'pl': 'Polish',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'cs': 'Czech',
            'hu': 'Hungarian',
        }
        # Only include supported languages
        supported = set(text_agent.language_manager.supported_languages)
        # Try to match both code and name
        languages = {code: name for code, name in language_map.items() if code in supported or name.lower() in supported}
        return {
            "success": True,
            "languages": languages,
            "count": len(languages)
        }
    except Exception as e:
        logger.error(f"Error getting text languages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available text languages: {str(e)}"
        )

@app.get("/languages-voice")
async def get_voice_languages():
    """
    Get list of available languages for voice synthesis using Coqui TTS XTTS v2.
    """
    try:
        # Get voice synthesis languages from XTTS v2
        voice_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'pl': 'Polish',
            'tr': 'Turkish',
            'ru': 'Russian',
            'nl': 'Dutch',
            'cs': 'Czech',
            'ar': 'Arabic',
            'zh': 'Chinese',
            'hu': 'Hungarian',
            'ko': 'Korean',
            'ja': 'Japanese',
            'hi': 'Hindi'
        }
        
        return {
            "success": True,
            "languages": voice_languages,
            "count": len(voice_languages)
        }
    except Exception as e:
        logger.error(f"Error getting voice languages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available voice languages: {str(e)}"
        )

@app.post("/text-query", response_model=TextResponse)
async def process_text_query(query: TextQuery):
    """
    Process a text query and return the response.
    """
    try:
        logger.info(f"Processing text query: {query.text[:100]}...")
        
        detected_language = None
        detected_confidence = None
        input_text = query.text
        
        # Detect language if requested
        if query.auto_detect_language:
            try:
                detected_language, detected_confidence = text_agent.language_manager.detect_language(query.text)
                if detected_language:
                    logger.info(f"Detected language: {detected_language} (confidence: {detected_confidence:.2f})")
                    
                    # If translate_input is True, translate to English before processing
                    if query.translate_input and detected_language.lower() != "english":
                        try:
                            translator = GoogleTranslator(source=detected_language, target='english')
                            input_text = translator.translate(query.text)
                            logger.info(f"Translated input to English: {input_text[:100]}...")
                        except Exception as e:
                            logger.warning(f"Failed to translate input: {e}")
            except Exception as e:
                logger.warning(f"Language detection error: {e}")
        
        # Use specified language if not auto-detecting or detection failed
        target_language = detected_language if query.auto_detect_language and detected_language else query.language
        
        # Validate and normalize the requested language
        is_valid, normalized_language = text_agent.language_manager.validate_language(target_language)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {target_language}. Use /languages-text endpoint to see available languages."
            )
            
        # Set the validated language
        text_agent.user_language = normalized_language
        logger.info(f"Language set to: {normalized_language}")
        
        # Generate response using potentially translated input text
        response = await text_agent.generate_response(input_text)
        
        # Translate response if not English
        if normalized_language.lower() != "english":
            logger.info(f"Translating response to {normalized_language}")
            try:
                response = await text_agent.enhanced_translator.translate_text(response, normalized_language)
                if not response:
                    raise ValueError("Translation failed")
            except Exception as e:
                logger.error(f"Translation error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to translate response to {normalized_language}"
                )
        
        return TextResponse(
            success=True,
            response=response,
            language=normalized_language,
            detected_language=detected_language,
            confidence=detected_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text query: {str(e)}"
        )

@app.post("/voice-query", response_model=VoiceResponse)
async def process_voice_query(
    audio_file: UploadFile = File(...),
    language: str = Form("english"),  # Accept language from FormData, default to English
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process a voice query and return both text and synthesized speech response.
    """
    try:
        logger.info(f"Processing voice query from file: {audio_file.filename}")
        logger.info(f"Selected language for TTS (raw): {language}")
        xtts_code = get_xtts_language_code(language)
        logger.info(f"XTTS v2 language code used: {xtts_code}")

        # Map code to DeepTranslator language name
        deep_translator_language = map_code_to_deep_translator_language(language)
        logger.info(f"DeepTranslator target language: {deep_translator_language}")

        # Validate language for agent
        is_valid, agent_language = text_agent.language_manager.validate_language(deep_translator_language)
        logger.info(f"Normalized agent language for translation: {agent_language} (valid: {is_valid})")
        if not is_valid:
            logger.error(f"Unsupported language for translation: {language}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {language}. Use /languages-voice endpoint to see available languages."
            )

        # Read uploaded audio file
        audio_data = await audio_file.read()
        logger.info(f"Received audio data: {len(audio_data)} bytes")

        # Validate and convert audio to WAV if needed
        wav_data = validate_and_convert_audio(audio_data)
        if not wav_data:
            logger.error("Invalid or unsupported audio format. Could not convert to WAV.")
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported audio format. Please send a valid audio file."
            )

        # Save to temporary file for processing
        temp_input = TEMP_DIR / f"input_{uuid.uuid4()}.wav"
        with open(temp_input, "wb") as f:
            f.write(wav_data)
        logger.info(f"Saved audio to temporary file: {temp_input}")

        try:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5
            with sr.AudioFile(str(temp_input)) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)  # type: ignore[attr-defined]
            if not text:
                logger.warning("No speech detected in audio")
                raise sr.UnknownValueError("No speech detected")
            logger.info(f"Successfully transcribed text: {text}")
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            # Fallback: generate TTS for error message and return with 200 OK
            fallback_text = "Sorry, I could not understand your voice. Please try again."
            # Translate the fallback text to the selected language
            text_agent.user_language = deep_translator_language
            try:
                translated_fallback_text = await text_agent.enhanced_translator.translate_text(fallback_text, deep_translator_language)
                logger.info(f"Translated fallback text to {deep_translator_language}: {translated_fallback_text}")
            except Exception as e:
                logger.error(f"Translation failed for fallback text to {deep_translator_language}: {e}")
                translated_fallback_text = fallback_text
            fallback_xtts_code = get_xtts_language_code(language)
            fallback_audio_data = voice_agent.text_to_speech.convert_to_speech(translated_fallback_text, language=fallback_xtts_code)
            if fallback_audio_data:
                temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.wav"
                with open(temp_output, "wb") as f:
                    f.write(fallback_audio_data)
                background_tasks.add_task(cleanup_old_files, TEMP_DIR)
                audio_url = f"/audio/{temp_output.name}"
                return VoiceResponse(
                    user_text=fallback_text,
                    response_text=translated_fallback_text,
                    audio_url=audio_url,
                    error="Could not understand audio input"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Could not understand audio input and TTS fallback failed."
                )
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Speech recognition service error: {str(e)}"
            )
        finally:
            if temp_input.exists():
                temp_input.unlink()

        # Always set the agent's user_language to the mapped DeepTranslator language
        text_agent.user_language = deep_translator_language
        response_text = await text_agent.generate_response(text)
        logger.info(f"Generated response (before translation): {response_text}")

        # Always translate the response to the selected language (even if it's English, for consistency)
        try:
            translated_response = await text_agent.enhanced_translator.translate_text(response_text, deep_translator_language)
            logger.info(f"Translated response to {deep_translator_language}: {translated_response}")
        except Exception as e:
            logger.error(f"Translation failed for {deep_translator_language}: {e}")
            translated_response = response_text

        # Synthesize speech in the selected language
        audio_data = voice_agent.text_to_speech.convert_to_speech(translated_response, language=xtts_code)
        if not audio_data:
            logger.warning("Speech synthesis failed, returning text-only response")
            return VoiceResponse(
                user_text=response_text,
                response_text=translated_response,
                error="Speech synthesis unavailable"
            )
        temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.wav"
        with open(temp_output, "wb") as f:
            f.write(audio_data)
        background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        audio_url = f"/audio/{temp_output.name}"
        logger.info("Successfully processed voice query and generated response")
        return VoiceResponse(
            user_text=response_text,
            response_text=translated_response,
            audio_url=audio_url,
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice query: {str(e)}"
        )

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve generated audio files.
    """
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint with enhanced status information.
    """
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0",
        "services": {
            "text_agent": "ok",
            "voice_agent": "ok",
            "translation": "ok"
        }
    }
    
    # Check text agent
    try:
        # Simple test to verify text agent is working
        _ = text_agent.language_manager.supported_languages
    except Exception as e:
        status["services"]["text_agent"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check voice agent
    try:
        # Simple test to verify voice agent is working
        _ = voice_agent.text_to_speech.supported_languages
    except Exception as e:
        status["services"]["voice_agent"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check translation service
    try:
        translator = GoogleTranslator(source='en', target='es')
        test_translation = translator.translate("hello")
        if not test_translation:
            raise ValueError("Empty translation result")
    except Exception as e:
        status["services"]["translation"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return status

@app.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts in a single request.
    """
    try:
        logger.info(f"Processing batch translation request: {len(request.texts)} texts to {request.target_language}")
        
        # Validate target language
        is_valid, normalized_language = text_agent.language_manager.validate_language(request.target_language)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target language: {request.target_language}. Use /languages-text endpoint to see available languages."
            )
        
        # Skip translation if source and target are the same
        if request.source_language.lower() == normalized_language.lower():
            return BatchTranslationResponse(
                success=True,
                translations=request.texts,
                target_language=normalized_language
            )
            
        # Process translations with caching
        translations = []
        failed_indices = []
        
        for i, text in enumerate(request.texts):
            try:
                if not text.strip():
                    translations.append("")
                    continue
                
                # Generate cache key
                cache_key = f"{hashlib.md5((text[:100] + request.source_language + normalized_language).encode()).hexdigest()}"
                
                # Check cache first
                if cache_key in batch_translation_cache:
                    translations.append(batch_translation_cache[cache_key])
                    logger.info(f"Cache hit for text at index {i}")
                    continue
                    
                # Prepare translator if not already cached
                if not hasattr(app.state, 'translators'):
                    app.state.translators = {}
                    
                translator_key = f"{request.source_language}_{normalized_language}"
                if translator_key not in app.state.translators:
                    try:
                        app.state.translators[translator_key] = GoogleTranslator(
                            source=request.source_language, 
                            target=normalized_language
                        )
                    except Exception as e:
                        logger.error(f"Failed to initialize translator: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to initialize translator: {str(e)}"
                        )
                
                translator = app.state.translators[translator_key]
                translated = translator.translate(text)
                
                if translated:
                    # Cache successful translation
                    batch_translation_cache[cache_key] = translated
                    translations.append(translated)
                else:
                    translations.append(text)
                    failed_indices.append(i)
                    
                # Add small delay to prevent API rate limiting
                if i < len(request.texts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Failed to translate text at index {i}: {e}")
                translations.append(text)  # Use original text as fallback
                failed_indices.append(i)
        
        return BatchTranslationResponse(
            success=True,
            translations=translations,
            target_language=normalized_language,
            failed_indices=failed_indices if failed_indices else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch translation: {e}")
        return BatchTranslationResponse(
            success=False,
            translations=[],
            target_language=request.target_language,
            error=str(e)
        )

@app.get("/translation-cache-stats")
async def get_translation_cache_stats():
    """
    Get statistics about the translation cache.
    """
    try:
        agent_cache = text_agent.translation_cache
        batch_cache = batch_translation_cache
        
        return {
            "success": True,
            "agent_cache": {
                "size": len(agent_cache),
                "max_size": agent_cache.maxsize,
                "ttl_seconds": agent_cache.ttl,
                "languages": list(set([key.split('_')[-1] for key in agent_cache.keys() if '_' in key]))
            },
            "batch_cache": {
                "size": len(batch_cache),
                "max_size": batch_cache.maxsize,
                "ttl_seconds": batch_cache.ttl
            },
            "total_cached_items": len(agent_cache) + len(batch_cache)
        }
    except Exception as e:
        logger.error(f"Error getting translation cache stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of the provided text.
    """
    try:
        if not request.text.strip():
            return LanguageDetectionResponse(
                detected_language="unknown",
                success=False,
                error="Empty text provided"
            )
            
        # Use the enhanced language detection from LanguageManager
        detected_language, confidence = text_agent.language_manager.detect_language(request.text)
        
        if not detected_language:
            return LanguageDetectionResponse(
                detected_language="unknown",
                success=False,
                error="Could not detect language"
            )
            
        return LanguageDetectionResponse(
            detected_language=detected_language,
            confidence=confidence,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return LanguageDetectionResponse(
            detected_language="unknown",
            success=False,
            error=str(e)
        )

@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def convert_text_to_speech(request: TextToSpeechRequest, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Convert text to speech using Coqui TTS and return audio URL.
    """
    try:
        # Clean and prepare text
        text = request.text.strip()
        
        # Ensure minimum text length for kernel size
        min_text_length = 50  # Minimum length to satisfy kernel size
        if len(text) < min_text_length:
            text = f"Hello. {text} Thank you for your question. I hope this helps."
            
        # Split text into sentences and ensure each chunk is long enough
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        text_chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # If a single sentence is too short, add padding
            if len(sentence) < min_text_length:
                sentence = f"Let me say that again. {sentence} Thank you."
            
            if current_length + len(sentence) > 200:  # Maximum chunk size
                if current_chunk:
                    text_chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            text_chunks.append(' '.join(current_chunk))
            
        logger.info(f"Converting text to speech: {text[:50]}...")
        logger.info(f"Split into {len(text_chunks)} chunks")
        
        # Process each chunk and combine audio
        all_audio_data = []
        for chunk in text_chunks:
            try:
                xtts_code = get_xtts_language_code(request.language or "english")
                audio_data = voice_agent.text_to_speech.convert_to_speech(chunk, language=xtts_code)
                if audio_data:
                    all_audio_data.append(audio_data)
            except Exception as chunk_error:
                logger.warning(f"Error processing chunk: {chunk_error}")
                padded_chunk = f"Let me repeat that. {chunk} Thank you for listening."
                try:
                    xtts_code = get_xtts_language_code(request.language or "english")
                    audio_data = voice_agent.text_to_speech.convert_to_speech(padded_chunk, language=xtts_code)
                    if audio_data:
                        all_audio_data.append(audio_data)
                except Exception as retry_error:
                    logger.error(f"Failed to process chunk even with padding: {retry_error}")
        
        if not all_audio_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to convert text to speech"
            )
            
        # Combine all audio chunks
        combined_audio = b''.join(all_audio_data)
            
        # Save audio to temporary file
        temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.wav"
        with open(temp_output, "wb") as f:
            f.write(combined_audio)
            
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
        # Generate URL for audio file
        audio_url = f"/audio/{temp_output.name}"
        
        return TextToSpeechResponse(
            success=True,
            audio_url=audio_url,
            text=text
        )
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error converting text to speech: {str(e)}"
        )

@app.post("/tts-stream")
async def tts_stream(text: str, language: str = "english"):
    """
    Stream TTS audio directly as audio/wav for browser playback.
    """
    try:
        xtts_code = get_xtts_language_code(language)
        audio_data = voice_agent.text_to_speech.convert_to_speech(text, language=xtts_code)
        if not audio_data:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS stream error: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "server2:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    ) 