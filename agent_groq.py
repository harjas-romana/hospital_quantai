"""
QuantAI Hospital Assistant (Auckland, New Zealand)
Advanced implementation integrating OpenRouter LLM for context-aware, multilingual hospital communication.
All logic, prompts, and responses are strictly focused on QuantAI Hospital's operations in Auckland, New Zealand.
"""

import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
from deep_translator import GoogleTranslator
from colorama import init, Fore, Style, Back
from tqdm import tqdm
import time
from cachetools import TTLCache, cached
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Any, Optional, Tuple
import re
import pickle
from pathlib import Path
import logging
from datetime import datetime
from fuzzywuzzy import fuzz, process
import asyncio
import groq
import rag_layer  # Import the QuantAI Hospital RAG system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quantai_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored output
init()

class LanguageManager:
    """Manages language selection and translation with advanced features and persistence."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.language_cache_file = self.cache_dir / "language_preferences.pkl"
        self.language_preferences = self._load_language_preferences()
        
        # Initialize supported languages
        try:
            # Get languages from Google Translator
            translator = GoogleTranslator(source='auto', target='en')
            self.supported_languages = set(translator.get_supported_languages())
        except Exception as e:
            logger.warning(f"Error getting supported languages: {e}")
            # Fallback to common languages if API fails
            self.supported_languages = {
                'english', 'spanish', 'french', 'german', 'italian', 'portuguese',
                'chinese', 'japanese', 'korean', 'arabic', 'russian', 'hindi'
            }
        
        self.language_aliases = self._initialize_language_aliases()
        
    def _initialize_language_aliases(self) -> Dict[str, str]:
        """Initialize common language aliases and variations."""
        return {
            "chinese": "chinese",
            "cn": "chinese",
            "zh": "chinese",
            "español": "spanish",
            "esp": "spanish",
            "français": "french",
            "fr": "french",
            "deutsch": "german",
            "de": "german",
            "italiano": "italian",
            "it": "italian",
            "português": "portuguese",
            "pt": "portuguese",
            "русский": "russian",
            "ru": "russian",
            "हिंदी": "hindi",
            "hi": "hindi",
            "日本語": "japanese",
            "ja": "japanese",
            "한국어": "korean",
            "ko": "korean",
            "العربية": "arabic",
            "ar": "arabic"
        }

    def _load_language_preferences(self) -> Dict[str, str]:
        """Load saved language preferences with error handling."""
        try:
            if self.language_cache_file.exists():
                with open(self.language_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load language preferences: {e}")
        return {}

    def save_language_preference(self, user_id: str, language: str):
        """Save user's language preference persistently."""
        self.language_preferences[user_id] = language
        try:
            with open(self.language_cache_file, 'wb') as f:
                pickle.dump(self.language_preferences, f)
        except Exception as e:
            logger.warning(f"Could not save language preference: {e}")

    def get_language_preference(self, user_id: str) -> Optional[str]:
        """Retrieve user's saved language preference."""
        return self.language_preferences.get(user_id)

    def validate_language(self, language: str) -> Tuple[bool, str]:
        """Validate and normalize language input."""
        language = language.lower().strip()
        
        # Direct match
        if language in self.supported_languages:
            return True, language
            
        # Check aliases
        if language in self.language_aliases:
            normalized = self.language_aliases[language]
            if normalized in self.supported_languages:
                return True, normalized
            
        # Fuzzy matching for close matches
        matches = process.extractBests(
            language,
            self.supported_languages,
            score_cutoff=80,
            limit=3
        )
        
        if matches:
            return True, matches[0][0]
            
        return False, ""

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of a given text using GoogleTranslator.
        
        Args:
            text: The text to detect language for
            
        Returns:
            Tuple containing (detected_language, confidence_score)
        """
        try:
            # Use GoogleTranslator to detect language
            translator = GoogleTranslator(source='auto', target='en')
            
            # The translator doesn't have a direct language detection method,
            # but we can use the source language from a translation
            translator.source = 'auto'
            translator.translate(text[:100])  # Use just a sample of text
            detected_code = translator._source
            
            # Map language code to language name
            language_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'zh-cn': 'chinese',
                'zh-tw': 'chinese',
                'zh': 'chinese',
                'ja': 'japanese',
                'ko': 'korean',
                'ar': 'arabic',
                'ru': 'russian',
                'hi': 'hindi'
            }
            
            detected_language = language_map.get(detected_code, detected_code)
            
            # Validate the detected language is in our supported languages
            is_valid, normalized_language = self.validate_language(detected_language)
            if is_valid:
                return normalized_language, 0.9  # Confidence score placeholder
            else:
                return detected_code, 0.7  # Return original code with lower confidence
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "english", 0.0  # Default to English with zero confidence

    def display_languages(self):
        """Display available languages in an organized, searchable format."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Available Languages {Style.RESET_ALL}")
        print("\nType part of a language name to search, or press Enter to see all languages.")
        
        while True:
            search = input(f"\n{Fore.YELLOW}Search languages (or press Enter): {Style.RESET_ALL}").lower()
            
            # Filter languages based on search term
            if search:
                matching_languages = [
                    lang for lang in sorted(self.supported_languages)
                    if search in lang.lower()
                ]
            else:
                matching_languages = sorted(self.supported_languages)
            
            if not matching_languages:
                print(f"{Fore.RED}No languages found matching '{search}'{Style.RESET_ALL}")
                continue
            
            # Display matching languages in columns
            col_width = 25
            num_cols = 3
            
            print(f"\n{Fore.CYAN}Matching languages:{Style.RESET_ALL}")
            for i in range(0, len(matching_languages), num_cols):
                row = matching_languages[i:i + num_cols]
                print("".join(f"{lang:<{col_width}}" for lang in row))
            
            selection = input(f"\n{Fore.YELLOW}Select a language (or type 'search' to search again): {Style.RESET_ALL}").lower()
            
            if selection == 'search':
                continue
                
            valid, normalized_language = self.validate_language(selection)
            if valid:
                return normalized_language
            else:
                print(f"{Fore.RED}Invalid language selection. Please try again.{Style.RESET_ALL}")

class ConversationContext:
    """Manages conversation context for enhanced follow-up query handling, entity tracking, and pronoun resolution."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
        self.last_entities = {}
        self.last_subject = None
        self.last_query_type = None
        self.last_doctor = None
        self.last_department = None
        self.last_patient = None
        self.last_topic = None
    
    def add_interaction(self, query: str, response: str, query_type: str = None, entities: dict = None):
        """Add a query-response pair to the conversation history and update tracked entities."""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now(),
            'query_type': query_type,
            'entities': entities or {}
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        # Update tracked entities
        if entities:
            if 'doctor' in entities:
                self.last_doctor = entities['doctor']
            if 'department' in entities:
                self.last_department = entities['department']
            if 'patient' in entities:
                self.last_patient = entities['patient']
            if 'topic' in entities:
                self.last_topic = entities['topic']
        self.last_entities = entities or {}
        self.last_query_type = query_type
    
    def get_relevant_context(self, current_query: str) -> str:
        """Get relevant context from conversation history for the current query."""
        context_parts = []
        if self.last_doctor:
            context_parts.append(f"Doctor: {self.last_doctor}")
        if self.last_department:
            context_parts.append(f"Department: {self.last_department}")
        if self.last_patient:
            context_parts.append(f"Patient: {self.last_patient}")
        if self.last_topic:
            context_parts.append(f"Topic: {self.last_topic}")
        if self.conversation_history:
            last = self.conversation_history[-1]
            context_parts.append(f"Previous Query: {last['query']}")
            context_parts.append(f"Previous Response: {last['response'][:100]}")
        return "\n".join(context_parts)
    
    def extract_entities(self, text: str) -> dict:
        """Extract key entities (doctor, department, patient, etc.) from text for context tracking."""
        entities = {}
        # Simple regex-based extraction for demo purposes
        doctor_match = re.search(r"Dr\.\s*([A-Za-z]+)", text)
        if doctor_match:
            entities['doctor'] = doctor_match.group(0)
        department_match = re.search(r"(cardiology|emergency|pediatrics|radiology|surgery|icu|ward)", text, re.I)
        if department_match:
            entities['department'] = department_match.group(0).capitalize()
        patient_match = re.search(r"patient\s*([A-Za-z]+)", text, re.I)
        if patient_match:
            entities['patient'] = patient_match.group(1)
        # Add more entity extraction as needed
        return entities
    
    def resolve_pronouns(self, query: str) -> str:
        """Replace pronouns in the query with the last known entity for clarity."""
        pronoun_map = {
            'he': self.last_doctor or self.last_patient,
            'she': self.last_doctor or self.last_patient,
            'they': self.last_doctor or self.last_patient,
            'him': self.last_doctor or self.last_patient,
            'her': self.last_doctor or self.last_patient,
            'doctor': self.last_doctor,
            'department': self.last_department,
            'patient': self.last_patient
        }
        for pronoun, entity in pronoun_map.items():
            if entity and re.search(rf'\b{pronoun}\b', query, re.I):
                query = re.sub(rf'\b{pronoun}\b', entity, query, flags=re.I)
        return query

class EnhancedTranslator:
    """Enhanced translation manager with caching and async support."""
    
    def __init__(self, cache_ttl: int = 3600):
        self.translation_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.language_cache = {}
        self.lock = asyncio.Lock()
        self.batch_size = 1000  # Characters per batch for chunked translation
        
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language with caching and chunking."""
        if not text or target_language == 'english':
            return text
            
        # Check cache first
        cache_key = f"{text[:50]}_{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        try:
            async with self.lock:  # Ensure thread safety
                # Split text into manageable chunks
                chunks = self._split_into_chunks(text)
                translated_chunks = []
                
                translator = GoogleTranslator(source='english', target=target_language)
                
                for chunk in chunks:
                    chunk_cache_key = f"{chunk}_{target_language}"
                    if chunk_cache_key in self.translation_cache:
                        translated_chunk = self.translation_cache[chunk_cache_key]
                    else:
                        translated_chunk = translator.translate(chunk)
                        if translated_chunk:
                            self.translation_cache[chunk_cache_key] = translated_chunk
                        else:
                            translated_chunk = chunk
                            
                    translated_chunks.append(translated_chunk)
                    
                final_translation = ' '.join(translated_chunks)
                self.translation_cache[cache_key] = final_translation
                return final_translation
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
            
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for efficient translation."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.batch_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages with caching."""
        if self.language_cache:
            return list(self.language_cache.keys())
            
        try:
            translator = GoogleTranslator(source='auto', target='en')
            languages = translator.get_supported_languages()
            self.language_cache = {lang: True for lang in languages}
            return languages
        except Exception as e:
            logger.error(f"Error getting supported languages: {e}")
            return ['english']  # Fallback to English only
            
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text with confidence score."""
        try:
            translator = GoogleTranslator(source='auto', target='en')
            detected = translator.detect(text)
            return detected.lang, detected.confidence
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'english', 0.0
            
    async def close(self):
        """Clean up resources."""
        # Nothing to clean up for now, but included for future-proofing
        pass

class QuantAIHospitalAgent:
    """Strictly domain-specific AI agent for QuantAI Hospital Assistant (Auckland, New Zealand).
    All responses and context are based on QuantAI Hospital's data and operations in Auckland.
    """
    
    def __init__(self):
        """Initialize the QuantAI Hospital Agent with optimized components."""
        # Load environment variables
        load_dotenv()
        self._validate_environment()
        
        # Initialize language management
        self.language_manager = LanguageManager()
        self.user_language = None
        
        # Initialize optimized caching system
        self._initialize_caches()
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Load and prepare hospital data
        print(f"{Fore.CYAN}Initializing QuantAI Hospital systems...{Style.RESET_ALL}")
        self.load_hospital_data()
        
        # Initialize API configuration
        self._initialize_api_config()
        
        # Prepare vectorized data for fast matching
        print(f"{Fore.CYAN}Building knowledge vectors...{Style.RESET_ALL}")
        self._prepare_vectorized_data()
        
        # Precompute common queries for faster responses
        self._precompute_common_contexts()
        
        # Initialize conversation context
        self.conversation_context = ConversationContext()
        
        # Initialize enhanced translation manager
        self.enhanced_translator = EnhancedTranslator()
        
        logger.info("QuantAI Hospital Agent initialized successfully")
        print(f"{Fore.GREEN}✓ QuantAI Hospital Assistant ready{Style.RESET_ALL}")

    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = ['GROQ_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required API keys: {', '.join(missing_vars)}")
        self.groq_api_key = os.getenv('GROQ_API_KEY')

    def _initialize_api_config(self):
        """Initialize API configuration for QuantAI Hospital Assistant (Auckland) using Groq."""
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_client = groq.AsyncGroq(api_key=self.groq_api_key)

    def _initialize_caches(self):
        """Initialize caching system for optimal performance."""
        self.response_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour for responses
        self.context_cache = TTLCache(maxsize=100, ttl=1800)   # 30 minutes for context
        self.translation_cache = TTLCache(maxsize=300, ttl=7200)  # 2 hours for translations
        self.similarity_cache = TTLCache(maxsize=100, ttl=3600)   # 1 hour for similarity

    def _initialize_nlp_components(self):
        """Initialize NLP components with fallback options."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('words', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                max_features=10000
            )
        except Exception as e:
            logger.warning(f"Error initializing NLP components: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
            self.vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))

    def _prepare_vectorized_data(self):
        """Prepare vectorized representations of the dataset contents for similarity matching."""
        self.dataset_vectors = {}
        for name, df in self.hospital_data.items():
            # Convert DataFrame to text for vectorization
            text_data = df.astype(str).agg(' '.join, axis=1).tolist()
            if text_data:
                self.dataset_vectors[name] = self.vectorizer.fit_transform(text_data)

    def load_hospital_data(self):
        """Load all CSV files from the data directory."""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            
            self.hospital_data = {}
            self.data_metadata = {}
            
            print(f"{Fore.CYAN}Loading data files...{Style.RESET_ALL}")
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    print(f"Reading {file}...")
                    df = pd.read_csv(file_path)
                    
                    # Store the data
                    dataset_name = file.replace('.csv', '').replace('quantai_hospital_', '')
                    self.hospital_data[dataset_name] = df
                    
                    # Store basic metadata
                    self.data_metadata[dataset_name] = {
                        'columns': list(df.columns),
                        'row_count': len(df)
                    }
            
            print(f"{Fore.GREEN}✓ Successfully loaded {len(self.hospital_data)} data files{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading data: {e}{Style.RESET_ALL}")
            raise

    def _build_rag_context(self, query: str) -> str:
        """Build context from loaded data files with hospital-specific information."""
        try:
            context_parts = []
            
            # Add QuantAI Hospital basic information
            context_parts.extend([
                "QuantAI Hospital Information:",
                "- Location: Auckland, New Zealand",
                "- Type: Full-service medical facility",
                "- Emergency Services: Available 24/7",
                "- Main Contact: +64 9 123 4567"
            ])
            
            # Add data summary
            context_parts.append("\nAvailable Hospital Records:")
            for name, metadata in self.data_metadata.items():
                context_parts.append(f"\n{name.replace('_', ' ').title()}:")
                context_parts.append(f"- {metadata['row_count']} records")
                context_parts.append(f"- Information includes: {', '.join(metadata['columns'])}")
            
            # Add relevant sample data
            context_parts.append("\nRelevant Records:")
            for name, df in self.hospital_data.items():
                if any(term in query.lower() for term in name.split('_')):
                    context_parts.append(f"\n{name.replace('_', ' ').title()}:")
                    sample = df.head(3).to_dict('records')
                    for record in sample:
                        record_str = ", ".join([f"{k}: {v}" for k, v in record.items() if pd.notna(v)])
                        context_parts.append(record_str)
            
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return "Basic QuantAI Hospital Information Available"

    async def generate_response(self, user_query: str) -> str:
        """Generate response using Groq API with loaded data context and follow-up awareness."""
        try:
            # Resolve pronouns/entities for follow-up
            resolved_query = self.conversation_context.resolve_pronouns(user_query)
            # Extract entities for context
            entities = self.conversation_context.extract_entities(resolved_query)
            # Build context from data and conversation
            context = self._build_rag_context(resolved_query)
            conversation_context = self.conversation_context.get_relevant_context(resolved_query)
            if conversation_context:
                context = f"{context}\n\nConversation Context:\n{conversation_context}"
            # Generate response using RAG system
            # Use Groq API directly for response generation
            system_prompt = (
                "You are the official AI assistant for QuantAI Hospital.\n"
                "- Greet users warmly and use natural, friendly, short sentences.\n"
                "- If you need more info, ask a brief follow-up question.\n"
                "- Always reference QuantAI Hospital's services, staff, and policies.\n"
                "- If info is missing, say: 'Sorry, I don't have that info right now.'\n"
                "- If asked about other hospitals, say: 'I'm here to help with QuantAI Hospital only.'\n"
                "- Keep answers brief, clear, and human.\n"
                "- Use a supportive, empathetic, and professional tone.\n"
                "- Use context and conversation history to resolve pronouns and follow-ups."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context (QuantAI Hospital data): {context}"},
                {"role": "user", "content": resolved_query}
            ]
            chat_completion = await self.groq_client.chat.completions.create(
                messages=messages,
                model=self.groq_model,
                temperature=0.7,
                max_tokens=350,
                top_p=0.9,
                presence_penalty=0.7,
                frequency_penalty=0.4
            )
            response = chat_completion.choices[0].message.content
            # Store in conversation history
            self.conversation_context.add_interaction(user_query, response, entities=entities)
            # Translate if needed
            if self.user_language and self.user_language != 'english':
                response = await self.enhanced_translator.translate_text(response, self.user_language)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request. Could you please rephrase?"

    async def run_cli(self):
        """Run the conversational command-line interface."""
        print(f"\n{Back.BLUE}{Fore.WHITE} QUANTAI HOSPITAL ASSISTANT {Style.RESET_ALL}")
        print("\nKia ora! I'm your QuantAI Hospital assistant. I'm here to help you with appointments,")
        print("patient information, services, and any other questions about our hospital.")
        
        # Select language
        await self.select_language()
        
        # Main interaction loop
        while True:
            print("\n" + "="*50)
            user_input = input(f"\n{Fore.CYAN}How can I assist you today? {Style.RESET_ALL}")
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n{Fore.CYAN}Thank you for using QuantAI Hospital's assistant. Take care and have a great day! {Style.RESET_ALL}")
                break
            
            if user_input.lower() in ['language', 'change language']:
                await self.change_language()
                continue
                
            if not user_input.strip():
                continue
            
            print(f"\n{Fore.CYAN}Let me help you with that...{Style.RESET_ALL}")
            
            try:
                # Create a subtle progress indicator
                with tqdm(total=100, desc="Processing", ncols=75, bar_format='{desc}: {percentage:3.0f}%|{bar}|') as pbar:
                    response = await self.generate_response(user_input)
                    pbar.update(100)
                
                # Print response with formatting
                print(f"\n{Fore.GREEN}QuantAI Hospital Assistant:{Style.RESET_ALL}")
                print(f"{response}")
                
                # Add a gentle prompt for follow-up
                print(f"\n{Fore.CYAN}Is there anything else you'd like to know about QuantAI Hospital?{Style.RESET_ALL}")
                
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n{Fore.RED}I apologize, but I'm having trouble accessing that information right now.{Style.RESET_ALL}")
                print("Could you please try asking your question again, or rephrase it?")

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for QuantAI Hospital context."""
        query_lower = query.lower()
        
        # Define QuantAI Hospital-specific query patterns
        patterns = {
            'patient': r'patient|illness|disease|treatment|health|admission|discharge|nhi',
            'staff': r'doctor|nurse|staff|specialist|consultant|surgeon|physician|pharmacist',
            'operations': r'operation|procedure|treatment|surgery|recovery|theatre|ward',
            'equipment': r'equipment|device|instrument|machine|scanner|monitor|ventilator',
            'services': r'service|clinic|outpatient|inpatient|specialist|referral|consultation',
            'emergency': r'emergency|urgent|immediate|critical|accident|trauma|triage',
            'location': r'location|address|directions|parking|auckland|transport|bus|train',
            'contact': r'contact|phone|email|fax|hours|opening|closing|waitangi|holiday',
            'departments': r'department|ward|icu|unit|clinic|radiology|cardiology|pediatrics',
            'appointments': r'appointment|booking|schedule|consultation|visit|checkup|follow-up',
            'facilities': r'facility|building|room|theatre|ward|clinic|centre|center'
        }
        
        # Check patterns
        for query_type, pattern in patterns.items():
            if re.search(pattern, query_lower):
                return query_type
                
        return 'general'

    def _precompute_common_contexts(self):
        """Precompute contexts for common query types to reduce latency."""
        self.precomputed_contexts = {}
        common_queries = [
            "patient information",
            "health statistics",
            "recent treatments",
            "equipment status",
            "emergency response",
            "community engagement",
            "cyberhealth operations"
        ]
        
        print(f"{Fore.CYAN}Precomputing knowledge contexts...{Style.RESET_ALL}")
        for query in common_queries:
            query_type = self._determine_query_type(query)
            context = []
            
            # Add metadata for relevant datasets
            for dataset_name, metadata in self.data_metadata.items():
                if self._is_dataset_relevant(dataset_name, query_type):
                    context.append(f"\n{dataset_name}:")
                    context.append(f"- Available fields: {', '.join(metadata['columns'])}")
                    context.append(f"- Total records: {metadata['row_count']}")
            
            self.precomputed_contexts[query_type] = "\n".join(context)
            
    def _is_dataset_relevant(self, dataset_name: str, query_type: str) -> bool:
        """Determine if a dataset is relevant to a query type."""
        relevance_map = {
            'patient': ['health_records', 'treatment_records', 'emergency_response'],
            'staff': ['doctors', 'nurses', 'medical_staff', 'healthcare_certifications'],
            'operations': ['surgery_logs', 'treatment_records', 'recovery_records'],
            'equipment': ['equipment_inventory'],
            'intelligence': ['research_reports', 'healthcare_data', 'cyberhealth_operations'],
            'emergency': ['emergency_response'],
            'community': ['community_engagement', 'indigenous_engagement'],
            'training': ['healthcare_certifications'],
            'cyber': ['cyberhealth_operations'],
            'maritime': ['maritime_operations'],
            'counter_terrorism': ['counter_terrorism_operations'],
            'indigenous': ['indigenous_engagement']
        }
        
        relevant_datasets = relevance_map.get(query_type, [])
        return dataset_name in relevant_datasets

    async def select_language(self):
        """Enhanced language selection with async support."""
        print(f"\n{Back.BLUE}{Fore.WHITE} Language Selection {Style.RESET_ALL}")
        print("\nPlease select your preferred language for communication.")
        
        # Get supported languages
        languages = await self.enhanced_translator.get_supported_languages()
        
        # Display languages in a user-friendly format
        print("\nAvailable languages:")
        for i, lang in enumerate(languages, 1):
            print(f"{i}. {lang.title()}")
            
        while True:
            try:
                choice = input(f"\n{Fore.YELLOW}Enter language number or name: {Style.RESET_ALL}")
                
                # Handle numeric choice
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(languages):
                        self.user_language = languages[idx]
                        break
                else:
                    # Handle text input
                    choice = choice.lower()
                    matches = [lang for lang in languages if lang.lower().startswith(choice)]
                    if len(matches) == 1:
                        self.user_language = matches[0]
                        break
                    elif len(matches) > 1:
                        print(f"\nMultiple matches found: {', '.join(matches)}")
                        print("Please be more specific.")
                    else:
                        print(f"\n{Fore.RED}Language not found. Please try again.{Style.RESET_ALL}")
            except ValueError:
                print(f"\n{Fore.RED}Invalid input. Please try again.{Style.RESET_ALL}")
                
        print(f"\n{Fore.GREEN}✓ Language set to: {self.user_language.title()}{Style.RESET_ALL}")
        
    async def change_language(self):
        """Change language preference with async support."""
        await self.select_language()
        response = "Language changed successfully. How may I assist you?"
        if self.user_language != 'english':
            response = await self.enhanced_translator.translate_text(response, self.user_language)
        print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL}")
        print(response)

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query with error handling."""
        try:
            # Basic tokenization
            tokens = word_tokenize(query.lower())
            tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
            
            try:
                # Try to use POS tagging if available
                pos_tags = nltk.pos_tag(tokens)
                important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                keywords = [word for word, pos in pos_tags if pos in important_pos]
                if keywords:
                    return keywords
            except Exception:
                # Fall back to basic filtering if POS tagging fails
                pass
            
            # Return all tokens if POS tagging failed or returned no keywords
            return tokens
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error in keyword extraction: {e}. Using basic filtering.{Style.RESET_ALL}")
            # Fallback to simple word splitting
            return [word.lower() for word in query.split() if word.lower() not in self.stop_words]

    def _find_relevant_data(self, query: str, df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
        """Find relevant rows in a DataFrame based on query similarity."""
        keywords = self._extract_query_keywords(query)
        
        # Convert DataFrame rows to text
        df_text = df.astype(str).agg(' '.join, axis=1)
        
        # Calculate similarity scores
        query_vector = self.vectorizer.transform([' '.join(keywords)])
        similarities = cosine_similarity(query_vector, self.vectorizer.transform(df_text))
        
        # Filter relevant rows
        relevant_indices = similarities[0] > threshold
        return df[relevant_indices]

    async def _prepare_context(self, query: str) -> str:
        """Prepare relevant context from hospital data based on the user query asynchronously."""
        try:
            # Check cache first for exact query
            cache_key = query[:50]
            if cache_key in self.context_cache:
                return self.context_cache[cache_key]
                
            # Get query type for context optimization
            query_type = self._determine_query_type(query)
            
            # Use precomputed context if available for this query type
            if query_type in self.precomputed_contexts:
                base_context = self.precomputed_contexts[query_type]
            else:
                # Generate base context from metadata
                context_parts = ["Hospital Data Overview:"]
                for dataset_name, metadata in self.data_metadata.items():
                    context_parts.append(f"\n{dataset_name}:")
                    context_parts.append(f"- Available fields: {', '.join(metadata['columns'])}")
                    context_parts.append(f"- Total records: {metadata['row_count']}")
                base_context = "\n".join(context_parts)
            
            # Extract keywords for targeted data retrieval
            keywords = self._extract_query_keywords(query)
            
            # Find and add relevant data from each dataset (limited to most relevant datasets)
            enhanced_context_parts = [base_context]
            relevant_datasets = []
            
            # Determine which datasets are most relevant to this query
            for dataset_name in self.hospital_data.keys():
                if self._is_dataset_relevant(dataset_name, query_type):
                    relevant_datasets.append(dataset_name)
            
            # If no specific datasets are identified, use a subset of all datasets
            if not relevant_datasets:
                relevant_datasets = list(self.hospital_data.keys())[:3]
                
            # Process only the most relevant datasets for faster response
            for dataset_name in relevant_datasets:
                try:
                    df = self.hospital_data[dataset_name]
                    relevant_data = self._find_relevant_data(query, df)
                    
                    if not relevant_data.empty:
                        enhanced_context_parts.append(f"\nRelevant {dataset_name} data:")
                        
                        # Add sample records (limited for speed)
                        enhanced_context_parts.append("\nSample records:")
                        sample = relevant_data.head(2).to_dict(orient='records')
                        enhanced_context_parts.append(json.dumps(sample, indent=2))
                        
                        # Add keyword matches (limited for speed)
                        for keyword in keywords[:3]:
                            matches = df.astype(str).apply(lambda x: x.str.contains(keyword, case=False)).any()
                            if matches.any():
                                columns = list(matches[matches].index)[:5]  # Limit to 5 columns
                                enhanced_context_parts.append(f"\nColumns containing '{keyword}': {', '.join(columns)}")
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_name}: {e}")
                    continue
            
            # Add conversation context
            conversation_context = self.conversation_context.get_relevant_context(query)
            if conversation_context:
                enhanced_context_parts.append("\nConversation Context:")
                enhanced_context_parts.append(conversation_context)
            
            final_context = "\n".join(enhanced_context_parts)
            
            # Cache the result
            self.context_cache[cache_key] = final_context
            return final_context
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return "Error preparing context. Using general hospital information."

async def main():
    """Main entry point for QuantAI Hospital Auckland Assistant."""
    try:
        print(f"\n{Back.BLUE}{Fore.WHITE} Initializing QuantAI Hospital Auckland Assistant {Style.RESET_ALL}")
        print("\nPreparing systems...")
        
        with tqdm(total=100, desc="Loading", ncols=75) as pbar:
            agent = QuantAIHospitalAgent()
            pbar.update(100)
        
        await agent.run_cli()
        
    except ValueError as e:
        print(f"\n{Back.RED}{Fore.WHITE} Configuration Error {Style.RESET_ALL}")
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        print("\nPlease ensure you have:")
        print("1. Valid API key in .env file")
        print("2. Required packages installed")
        print("3. Access to QuantAI Hospital data files")
        
    except Exception as e:
        print(f"\n{Back.RED}{Fore.WHITE} System Error {Style.RESET_ALL}")
        print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        logger.error(f"System error: {e}", exc_info=True)
        print("\nPlease try:")
        print("1. Checking your internet connection")
        print("2. Verifying system requirements")
        print("3. Contacting QuantAI Hospital IT support if needed")
    finally:
        if 'agent' in locals():
            if hasattr(agent, 'enhanced_translator'):
                await agent.enhanced_translator.close()
            print("\nQuantAI Hospital Assistant shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main()) 