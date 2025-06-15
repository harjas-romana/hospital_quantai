"""
QuantAI Hospital API Server
This module provides a FastAPI server that integrates both text and voice processing capabilities
from the QuantAI Hospital AI Assistant system.
"""

import os
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import tempfile
import uuid
from datetime import datetime
import speech_recognition as sr
import io
import wave
import time
import hashlib

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from deep_translator.exceptions import LanguageNotSupportedException
from cachetools import TTLCache

# Import our agent implementations
from agent import QuantAIAgent
from voice_agent import VoiceAgent, TextToSpeech, SpeechToText

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

# Initialize FastAPI app
app = FastAPI(
    title="QuantAI Hospital API",
    description="API server for QuantAI Hospital's text and voice processing capabilities",
    version="1.0.0"
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
text_agent = QuantAIAgent()
voice_agent = VoiceAgent()

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Create batch translation cache
batch_translation_cache = TTLCache(maxsize=500, ttl=3600)  # Cache for 1 hour

class TextQuery(BaseModel):
    """Model for text query requests."""
    text: str
    language: Optional[str] = "english"
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
    source_language: Optional[str] = "english"

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
    text: str
    audio_url: Optional[str] = None
    error: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    """Model for text-to-speech requests."""
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID by default

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

def validate_wav_file(audio_data: bytes) -> bool:
    """Validate if the audio data is a valid WAV file."""
    try:
        with io.BytesIO(audio_data) as f:
            # Check if it starts with RIFF header
            if audio_data[:4] != b'RIFF':
                logger.warning("Audio file does not start with RIFF header")
                return False
                
            # Try to open with wave module
            with wave.open(f, 'rb') as wav:
                # Get basic properties
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                
                logger.info(f"WAV file validated: channels={channels}, sample_width={sample_width}, framerate={framerate}")
                return True
    except Exception as e:
        logger.error(f"Error validating WAV file: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup."""
    logger.info("Starting QuantAI Hospital API Server")
    cleanup_old_files(TEMP_DIR)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down QuantAI Hospital API Server")
    cleanup_old_files(TEMP_DIR)

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
                detail=f"Unsupported language: {target_language}. Use /languages endpoint to see available languages."
            )
            
        # Set the validated language
        text_agent.user_language = normalized_language
        logger.info(f"Language set to: {normalized_language}")
        
        # Generate response using potentially translated input text
        response = text_agent.generate_response(input_text)
        
        # Translate response if not English
        if normalized_language.lower() != "english":
            logger.info(f"Translating response to {normalized_language}")
            try:
                response = text_agent.translate_text(response)
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
    background_tasks: BackgroundTasks = None
):
    """
    Process a voice query and return both text and synthesized speech response.
    """
    try:
        logger.info(f"Processing voice query from file: {audio_file.filename}")
        
        # Read uploaded audio file
        audio_data = await audio_file.read()
        logger.info(f"Received audio data: {len(audio_data)} bytes")
        
        # Validate WAV format
        if not validate_wav_file(audio_data):
            logger.error("Invalid WAV file format")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Please send a valid WAV file."
            )
            
        # Save to temporary file for processing
        temp_input = TEMP_DIR / f"input_{uuid.uuid4()}.wav"
        with open(temp_input, "wb") as f:
            f.write(audio_data)
            
        logger.info(f"Saved audio to temporary file: {temp_input}")
        
        try:
            # Create a recognizer with the same parameters as voice_agent.py
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5
            
            # Use AudioFile instead of AudioData for better compatibility
            with sr.AudioFile(str(temp_input)) as source:
                audio = recognizer.record(source)
                
            # Attempt recognition
            text = recognizer.recognize_google(audio)
            
            if not text:
                logger.warning("No speech detected in audio")
                raise sr.UnknownValueError("No speech detected")
                
            logger.info(f"Successfully transcribed text: {text}")
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            raise HTTPException(
                status_code=400,
                detail="Could not understand audio input"
            )
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Speech recognition service error: {str(e)}"
            )
        finally:
            # Clean up temporary input file
            if temp_input.exists():
                temp_input.unlink()
            
        # Generate response using the text agent
        response_text = text_agent.generate_response(text)
        logger.info(f"Generated response: {response_text}")
        
        # Convert response to speech using TTS
        audio_data = voice_agent.text_to_speech.convert_to_speech(response_text)
        
        if not audio_data:
            # Return text-only response if speech conversion fails
            logger.warning("Speech synthesis failed, returning text-only response")
            return VoiceResponse(
                text=response_text,
                error="Speech synthesis unavailable"
            )
            
        # Save audio response to temporary file
        temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.mp3"
        with open(temp_output, "wb") as f:
            f.write(audio_data)
            
        # Schedule cleanup of temporary files
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
        # Generate URL for audio file
        audio_url = f"/audio/{temp_output.name}"
        
        logger.info("Successfully processed voice query and generated response")
        return VoiceResponse(
            text=response_text,
            audio_url=audio_url
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
        media_type="audio/mpeg",
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
        _ = voice_agent.text_to_speech.api_key
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

@app.get("/languages")
async def get_available_languages():
    """
    Get list of available languages for translation.
    """
    try:
        languages = sorted(text_agent.language_manager.supported_languages)
        return {
            "success": True,
            "languages": languages,
            "count": len(languages)
        }
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available languages: {str(e)}"
        )

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
                detail=f"Unsupported target language: {request.target_language}. Use /languages endpoint to see available languages."
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

@app.post("/text-to-speech")
async def convert_text_to_speech(request: TextToSpeechRequest, background_tasks: BackgroundTasks = None):
    """
    Convert text to speech using ElevenLabs API and return audio URL.
    """
    try:
        logger.info(f"Converting text to speech: {request.text[:50]}...")
        
        # Convert text to speech
        audio_data = voice_agent.text_to_speech.convert_to_speech(request.text)
        
        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to convert text to speech"
            )
            
        # Save audio to temporary file
        temp_output = TEMP_DIR / f"output_{uuid.uuid4()}.mp3"
        with open(temp_output, "wb") as f:
            f.write(audio_data)
            
        # Schedule cleanup of temporary files
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, TEMP_DIR)
        
        # Generate URL for audio file
        audio_url = f"/audio/{temp_output.name}"
        
        return {
            "success": True,
            "audio_url": audio_url
        }
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error converting text to speech: {str(e)}"
        )

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
        "server:app",
        host="0.0.0.0",
        port=8000,  # Changed port from 8000 to 8080
        reload=True,
        log_level="info"
    ) 