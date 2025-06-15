"""
RAG (Retrieval-Augmented Generation) Layer for QuantAI Hospital
This module implements an advanced RAG system for healthcare data using QWEN AI via OpenRouter.
Features:
- Advanced NLP for query understanding
- Robust data cleaning and preprocessing
- Context-aware response generation
- HIPAA-compliant data handling
- Real-time resource management
- Synthetic data integration
- User intent analysis
- Sentiment analysis
- Security verification
- Empathetic responses
- Structured formatting
- Contextual memory
- Direct and clear responses
- Engaging follow-up questions
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import aiohttp
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
import jwt
from functools import lru_cache
import re
import time
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from fuzzywuzzy import fuzz
import concurrent.futures
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_layer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and manages synthetic data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data_dir = "data"
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load all synthetic data files."""
        try:
            # Load hospital infrastructure
            with open(os.path.join(self.data_dir, "hospital_infrastructure.json"), 'r') as f:
                self.data['infrastructure'] = json.load(f)
                
            # Load CSV files
            csv_files = [
                'quantai_hospital_patients.csv',
                'quantai_hospital_appointments.csv',
                'quantai_hospital_medical_history.csv',
                'quantai_hospital_staff_schedule.csv',
                'quantai_hospital_inventory_management.csv',
                'quantai_hospital_equipment_maintenance.csv'
            ]
            
            for file in csv_files:
                name = file.replace('quantai_hospital_', '').replace('.csv', '')
                self.data[name] = pd.read_csv(os.path.join(self.data_dir, file))
                
            logger.info("Successfully loaded all synthetic data")
            
        except Exception as e:
            logger.error(f"Error loading synthetic data: {str(e)}")
            raise
            
    def get_relevant_data(self, query_type: str, entities: List[str]) -> Dict[str, Any]:
        """Get relevant data based on query type and entities."""
        try:
            relevant_data = {}
            
            # Map query types to relevant data sources
            query_data_map = {
                'appointment': ['appointments', 'staff_schedule'],
                'medical_info': ['medical_history', 'patients'],
                'inventory': ['inventory_management'],
                'equipment': ['equipment_maintenance'],
                'staff': ['staff_schedule'],
                'general': ['infrastructure']
            }
            
            # Get relevant data sources
            data_sources = query_data_map.get(query_type, ['infrastructure'])
            
            # Extract relevant information
            for source in data_sources:
                if source in self.data:
                    if isinstance(self.data[source], pd.DataFrame):
                        # Filter DataFrame based on entities
                        filtered_data = self.data[source]
                        for entity in entities:
                            for col in filtered_data.columns:
                                if entity.lower() in filtered_data[col].astype(str).str.lower().values:
                                    filtered_data = filtered_data[filtered_data[col].astype(str).str.lower().str.contains(entity.lower())]
                        relevant_data[source] = filtered_data.to_dict('records')
                    else:
                        relevant_data[source] = self.data[source]
                        
            return relevant_data
            
        except Exception as e:
            logger.error(f"Error getting relevant data: {str(e)}")
            return {}

class QWENClient:
    """Client for interacting with QWEN AI via OpenRouter."""
    
    def __init__(self, api_key: str):
        """Initialize the QWEN client."""
        self.api_key = api_key
        self.session = None
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "qwen/qwen-2.5-7b-instruct:free"  # Using QWEN 72B model
        self.rate_limit = 10  # requests per minute
        self.last_request_time = 0
        
    async def initialize(self):
        """Initialize the aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://quantai-hospital.com",
                    "X-Title": "QuantAI Hospital Assistant"
                }
            )
            
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < (60 / self.rate_limit):
            await asyncio.sleep((60 / self.rate_limit) - time_since_last_request)
        self.last_request_time = time.time()
        
    async def generate_response(self, prompt: str, context: Optional[str] = None, user_role: str = "patient") -> str:
        """Generate a response using QWEN AI via OpenRouter."""
        try:
            await self._rate_limit()
            
            if not self.session:
                await self.initialize()
                
            # Construct the system prompt
            system_prompt = """You are a specialized AI assistant for QuantAI Hospital. Your responses should be:
            1. Specific to QuantAI Hospital's services, policies, and procedures
            2. Clear, concise, and professional
            3. Focused on healthcare and medical information
            4. HIPAA compliant and privacy-conscious
            5. Helpful and empathetic
            6. Based on the provided context and conversation history
            
            Do not provide information about other hospitals or general medical advice not specific to QuantAI Hospital."""
            
            # Construct the messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
                
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # Make the API request
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise Exception(f"API request failed with status {response.status}")
                    
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

class DataCleaner:
    """Handles data cleaning and preprocessing for healthcare data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using spaCy."""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

class QueryProcessor:
    """Processes and understands user queries."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.cleaner = DataCleaner()
        self.similarity_threshold = 0.7
        self.vectorizer = TfidfVectorizer()
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process and understand the user query."""
        try:
            # Clean the query
            cleaned_query = self.cleaner.clean_text(query)
            
            # Extract entities
            entities = self.cleaner.extract_entities(query)
            
            # Determine query type
            query_type = self._determine_query_type(cleaned_query)
            
            # Extract key information
            key_info = self._extract_key_info(cleaned_query, entities)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(cleaned_query)
            
            # Determine user intent
            intent = self._determine_intent(cleaned_query)
            
            return {
                'original_query': query,
                'cleaned_query': cleaned_query,
                'entities': entities,
                'query_type': query_type,
                'key_information': key_info,
                'sentiment': sentiment,
                'intent': intent
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'original_query': query,
                'cleaned_query': '',
                'entities': [],
                'query_type': 'error',
                'key_information': {'urgency_level': 'normal', 'requires_verification': False},
                'sentiment': 'neutral',
                'intent': 'unknown'
            }
            
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query_types = {
            'patient_info': ['patient', 'medical record', 'history'],
            'appointment': ['appointment', 'schedule', 'booking'],
            'department': ['department', 'ward', 'unit'],
            'emergency': ['emergency', 'urgent', 'critical'],
            'general': ['help', 'information', 'question']
        }
        
        for qtype, keywords in query_types.items():
            if any(keyword in query.lower() for keyword in keywords):
                return qtype
        return 'general'
        
    def _extract_key_info(self, query: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information from the query."""
        try:
            info = {
                'entities': entities,
                'timestamp': datetime.now().isoformat(),
                'urgency_level': 'normal',
                'requires_verification': False
            }
            
            # Check for urgency
            if any(word in query.lower() for word in ['emergency', 'urgent', 'immediate', 'critical']):
                info['urgency_level'] = 'high'
                
            # Check if verification is needed
            sensitive_patterns = [
                r'medical history',
                r'patient record',
                r'personal information',
                r'health data'
            ]
            
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in sensitive_patterns):
                info['requires_verification'] = True
                
            return info
            
        except Exception as e:
            logger.error(f"Error extracting key info: {str(e)}")
            return {
                'urgency_level': 'normal',
                'requires_verification': False
            }

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the text."""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 'neutral'
            
    def _determine_intent(self, query: str) -> str:
        """Determine the user's intent."""
        try:
            intent_patterns = {
                'request_info': r'what|how|when|where|why|tell me|explain',
                'make_request': r'need|want|would like|please|can you|could you',
                'confirm': r'is|are|do|does|did|have|has',
                'express_concern': r'worried|concerned|anxious|scared|afraid',
                'express_gratitude': r'thank|thanks|appreciate|grateful'
            }
            
            for intent, pattern in intent_patterns.items():
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
                    
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error determining intent: {str(e)}")
            return 'unknown'

class HospitalRAG:
    """Main RAG system for QuantAI Hospital."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.load_environment()
        self.initialize_components()
        self.conversation_history = []
        self.sentiment_history = []
        self.intent_history = []
        self.follow_up_questions = {
            'appointment': [
                "Would you like to schedule an appointment?",
                "Do you need help finding a suitable time slot?",
                "Would you like to know about our appointment policies?"
            ],
            'medical_info': [
                "Would you like more specific information about this?",
                "Do you have any other questions about this condition?",
                "Would you like to know about treatment options?"
            ],
            'emergency': [
                "Do you need immediate medical attention?",
                "Would you like me to connect you with emergency services?",
                "Are you experiencing any other symptoms?"
            ],
            'general': [
                "Is there anything specific you'd like to know?",
                "Would you like more details about this?",
                "Do you have any other questions?"
            ]
        }
        
    def load_environment(self):
        """Load environment variables."""
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
    def initialize_components(self):
        """Initialize system components."""
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.query_processor = QueryProcessor()
        self.qwen_client = QWENClient(self.api_key)
        
    async def process_query(self, query: str, user_role: str = "patient") -> str:
        """Process a query and generate a response."""
        try:
            # Clean and process the query
            cleaned_query = self.data_cleaner.clean_text(query)
            processed_query = self.query_processor.process_query(cleaned_query)
            
            # Get relevant data
            relevant_data = self.data_loader.get_relevant_data(
                processed_query['query_type'],
                [entity['text'] for entity in processed_query['entities']]
            )
            
            # Generate context
            context = self._generate_context(processed_query, user_role, relevant_data)
            
            # Generate response using Qwen AI
            response = await self.qwen_client.generate_response(
                prompt=query,
                context=context,
                user_role=user_role
            )
            
            # Format and clean the response
            formatted_response = self._format_response(response, processed_query)
            cleaned_response = self._clean_response(formatted_response)
            
            # Log the interaction
            self._log_interaction(query, cleaned_response, user_role)
            
            # Update conversation history
            self._update_conversation_history(query, cleaned_response, processed_query)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
            
    def _generate_context(self, processed_query: Dict[str, Any], user_role: str, relevant_data: Dict[str, Any]) -> str:
        """Generate context for the query."""
        context_parts = [
            f"You are a healthcare assistant at QuantAI Hospital.",
            f"The user is a {user_role}.",
            f"Query type: {processed_query['query_type']}",
            f"Urgency level: {processed_query['key_information'].get('urgency_level', 'normal')}",
            f"User intent: {processed_query['intent']}",
            f"Sentiment: {processed_query['sentiment']}"
        ]
        
        # Add conversation history context
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 interactions
            history_context = " | ".join([f"Previous: {h['query']} -> {h['response']}" for h in recent_history])
            context_parts.append(f"Recent conversation: {history_context}")
            
        # Add sentiment trend
        if self.sentiment_history:
            recent_sentiment = self.sentiment_history[-1]
            context_parts.append(f"Current sentiment trend: {recent_sentiment}")
            
        # Add intent trend
        if self.intent_history:
            recent_intent = self.intent_history[-1]
            context_parts.append(f"Current intent trend: {recent_intent}")
            
        if processed_query['entities']:
            context_parts.append(f"Key entities: {', '.join([e['text'] for e in processed_query['entities']])}")
            
        # Add relevant data context
        for source, data in relevant_data.items():
            if data:
                context_parts.append(f"Relevant {source} data available")
                
        return " | ".join(context_parts)
        
    def _log_interaction(self, query: str, response: str, user_role: str):
        """Log the interaction for monitoring and analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_role': user_role,
            'query': query,
            'response': response
        }
        logger.info(f"Interaction logged: {json.dumps(log_entry)}")
        
    def _format_response(self, response: str, processed_query: Dict[str, Any]) -> str:
        """Format the response based on query type and sentiment."""
        try:
            # Clean and structure the response
            response = self._clean_response(response)
            
            # Add empathetic prefix based on sentiment
            sentiment = processed_query['sentiment']
            if sentiment == 'negative':
                prefix = "I understand this might be concerning. "
            elif sentiment == 'positive':
                prefix = "I'm glad to help! "
            else:
                prefix = ""
                
            # Add structure based on query type
            query_type = processed_query['query_type']
            
            # Get relevant follow-up question
            follow_up = self._get_follow_up_question(query_type, processed_query)
            
            # Format based on query type
            if query_type == 'emergency':
                formatted = f"{prefix}🚨 {response}\n\n{follow_up}\n\nPlease note: If this is a medical emergency, call emergency services immediately."
            elif query_type == 'appointment':
                formatted = f"{prefix}📅 {response}\n\n{follow_up}"
            elif query_type == 'medical_info':
                formatted = f"{prefix}🏥 {response}\n\n{follow_up}"
            else:
                formatted = f"{prefix}{response}\n\n{follow_up}"
                
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return response
            
    def _clean_response(self, response: str) -> str:
        """Clean and structure the response for clarity."""
        try:
            # Remove unnecessary filler phrases
            filler_phrases = [
                "I understand that you're asking about",
                "I can tell you that",
                "Let me inform you that",
                "I would like to let you know that",
                "I'm happy to tell you that"
            ]
            
            for phrase in filler_phrases:
                response = response.replace(phrase, "")
                
            # Remove redundant phrases
            response = response.replace("  ", " ").strip()
            
            # Ensure response starts with a capital letter
            if response and not response[0].isupper():
                response = response[0].upper() + response[1:]
                
            return response
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return response
            
    def _get_follow_up_question(self, query_type: str, processed_query: Dict[str, Any]) -> str:
        """Get a relevant follow-up question based on query type and context."""
        try:
            # Get available follow-up questions for the query type
            questions = self.follow_up_questions.get(query_type, self.follow_up_questions['general'])
            
            # Check conversation history for previously asked questions
            asked_questions = set()
            for interaction in self.conversation_history[-3:]:
                if 'response' in interaction:
                    for question in questions:
                        if question in interaction['response']:
                            asked_questions.add(question)
                            
            # Filter out previously asked questions
            available_questions = [q for q in questions if q not in asked_questions]
            
            # If all questions have been asked, use a general one
            if not available_questions:
                available_questions = self.follow_up_questions['general']
                
            # Select a random question from available ones
            return random.choice(available_questions)
            
        except Exception as e:
            logger.error(f"Error getting follow-up question: {str(e)}")
            return "Is there anything else you'd like to know?"
            
    def _update_conversation_history(self, query: str, response: str, processed_query: Dict[str, Any]):
        """Update conversation history and trends."""
        try:
            # Add to conversation history
            self.conversation_history.append({
                'query': query,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update sentiment history
            self.sentiment_history.append(processed_query['sentiment'])
            
            # Update intent history
            self.intent_history.append(processed_query['intent'])
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)
            if len(self.sentiment_history) > 10:
                self.sentiment_history.pop(0)
            if len(self.intent_history) > 10:
                self.intent_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error updating conversation history: {str(e)}")
            
    async def close(self):
        """Close the RAG system."""
        await self.qwen_client.close()

# Initialize the RAG system
rag_system = HospitalRAG()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords') 