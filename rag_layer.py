"""
QuantAI Hospital Assistant (Auckland, New Zealand)
This module implements a strictly domain-specific RAG system for QuantAI Hospital in Auckland.
All responses are based solely on QuantAI Hospital's datasets and operations in Auckland, New Zealand.
Features:
- Context-aware response generation using OpenRouter (LLM) with strict QuantAI Hospital focus
- Auckland-specific knowledge integration (patients, appointments, staff, equipment, inventory, infrastructure)
- Concise, accurate, and contextually rich response formatting
- No speculation or general advice – only respond based on QuantAI Hospital's data
- All prompts and context reference QuantAI Hospital in Auckland
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import aiohttp
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_layer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HospitalDataLoader:
    """Loads and manages QuantAI Hospital datasets with strict New Zealand context and terminology."""

    def __init__(self):
        self.data_dir = "data"
        self.data: Dict[str, Any] = {}
        self.load_data()

    def load_data(self):
        """Load QuantAI hospital infrastructure and datasets for New Zealand context only."""
        try:
            infra_path = os.path.join(self.data_dir, "hospital_infrastructure.json")
            if os.path.exists(infra_path):
                with open(infra_path, "r") as f:
                    self.data["infrastructure"] = json.load(f)
            else:
                logger.warning("hospital_infrastructure.json not found in data directory")

            csv_files = [f for f in os.listdir(self.data_dir) if f.startswith("quantai_hospital_") and f.endswith(".csv")]

            for file in csv_files:
                dataset_name = file.replace("quantai_hospital_", "").replace(".csv", "")
                path = os.path.join(self.data_dir, file)
                try:
                    self.data[dataset_name] = pd.read_csv(path)
                    logger.info(f"Loaded {file}")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")

            if not self.data:
                logger.warning("No QuantAI hospital datasets loaded – verify /data directory")
        except Exception as e:
            logger.error(f"HospitalDataLoader error: {e}")
            raise

    def get_relevant_data(self, query_type: str, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve only New Zealand hospital data relevant to the query type and keywords."""
        try:
            # Map query types to relevant data sources (based on synthetic_dataset.py)
            mapping = {
                "patient": ["patients", "medical_history"],
                "appointment": ["appointments", "patients", "staff_schedule"],
                "staff": ["staff_schedule"],
                "equipment": ["equipment_maintenance", "inventory_management"],
                "inventory": ["inventory_management"],
                "infrastructure": ["infrastructure"],
                "medical": ["medical_history", "patients"],
                "history": ["medical_history"],
                "schedule": ["staff_schedule", "appointments"],
                "visit": ["appointments", "patients"],
                "general": ["infrastructure"]
            }
            data_sources = mapping.get(query_type, ["infrastructure"])
            relevant_data = {}

            for source in data_sources:
                if source in self.data:
                    df = self.data[source]
                    filtered_data = df

                    # Filter based on keywords (AND logic for specificity)
                    if keywords:
                        combined_mask = None
                        for keyword in keywords:
                            keyword_mask = None
                            for col in df.columns:
                                col_mask = df[col].astype(str).str.contains(
                                    keyword, case=False, na=False, regex=True
                                )
                                if keyword_mask is None:
                                    keyword_mask = col_mask
                                else:
                                    keyword_mask |= col_mask
                            if combined_mask is None:
                                combined_mask = keyword_mask
                            else:
                                combined_mask &= keyword_mask
                        if combined_mask is not None:
                            filtered_data = filtered_data[combined_mask]

                    # Sort by date if available
                    date_columns = [col for col in filtered_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_columns:
                        try:
                            filtered_data = filtered_data.sort_values(by=date_columns[0], ascending=False)
                        except Exception:
                            pass

                    if not filtered_data.empty:
                        relevant_data[source] = filtered_data.head(3).to_dict('records')

            return relevant_data
        except Exception as e:
            logger.error(f"Error getting relevant hospital data: {e}")
            return {}

class NZOpenRouterClient:
    """Client for interacting with OpenRouter LLM for QuantAI Hospital (Auckland) Assistant."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "meta-llama/llama-3.3-8b-instruct:free"
    
    async def initialize(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://quantai.co.nz",
                    "X-Title": "QuantAI Hospital Assistant"
                }
            )
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate_response(self, query: str, context: str, conversation_context: str = "", response_type: str = "standard") -> str:
        """
        Generate a brief, empathetic, and conversational response using OpenRouter, with follow-up and context awareness.
        """
        try:
            if not self.session:
                await self.initialize()
            # Conversational, brief, and follow-up aware system prompts
            system_prompts = {
                "standard": (
                    "You are the official AI assistant for QuantAI Hospital."
                    "\n- Greet users warmly and use natural, friendly, short sentences."
                    "\n- If you need more info, ask a brief follow-up question."
                    "\n- Always reference QuantAI Hospital's services, staff, and policies."
                    "\n- If info is missing, say: 'Sorry, I don't have that info right now.'"
                    "\n- If asked about other hospitals, say: 'I'm here to help with QuantAI Hospital only.'"
                    "\n- Keep answers brief, clear, and human."
                    "\n- Use a supportive, empathetic, and professional tone."
                    "\n- Use context and conversation history to resolve pronouns and follow-ups."
                ),
                "operational": (
                    "You are QuantAI Hospital's assistant (Operational Mode)."
                    "\n- Give short, actionable info in a friendly way."
                    "\n- Use bullet points if needed, but keep it conversational."
                ),
                "analytical": (
                    "You are QuantAI Hospital's assistant (Analytical Mode)."
                    "\n- Give brief, clear insights about hospital data."
                    "\n- Use a friendly, human tone."
                )
            }
            messages = [
                {"role": "system", "content": system_prompts.get(response_type, system_prompts["standard"])}
            ]
            if conversation_context:
                messages.append({"role": "system", "content": f"Conversation context: {conversation_context}"})
            messages.append({"role": "system", "content": f"Context (QuantAI Hospital data): {context}"})
            messages.append({"role": "user", "content": query})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 350,
                "top_p": 0.9,
                "presence_penalty": 0.7,
                "frequency_penalty": 0.4
            }
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

class HospitalAssistantRAG:
    """RAG system strictly for QuantAI Hospital Assistant (New Zealand context only)."""

    def __init__(self):
        self.data_loader = None
        self.gemma_client = None
        self.load_environment()
        self.initialize_components()
        # Optionally, define staff roles for access control if needed
        self.roles = [
            "Nurse", "Doctor", "Surgeon", "Administrator", "Receptionist", "Technician", "Pharmacist", "Manager"
        ]

    def load_environment(self):
        load_dotenv()
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    def initialize_components(self):
        self.data_loader = HospitalDataLoader()
        self.gemma_client = NZOpenRouterClient(self.api_key)

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for QuantAI Hospital Auckland context."""
        q = query.lower()
        patterns = {
            "patient": r"patient|demographic|medical history|diagnosis|condition|nhi|admission|discharge",
            "appointment": r"appointment|booking|visit|schedule|consultation|meeting",
            "staff": r"staff|doctor|nurse|surgeon|technician|physician|employee|specialist|consultant",
            "equipment": r"equipment|device|machine|maintenance|repair|malfunction|asset",
            "inventory": r"inventory|stock|supply|pharmacy|medication|consumable",
            "department": r"department|ward|icu|unit|clinic|radiology|cardiology|pediatrics|emergency",
            "medical": r"medication|treatment|therapy|lab|test|result|procedure|scan",
            "emergency": r"emergency|urgent|critical|triage|code|immediate",
            "location": r"location|address|directions|parking|auckland|transport|bus|train",
            "contact": r"contact|phone|email|fax|hours|opening|closing|waitangi|holiday",
            "services": r"service|specialist|consultation|referral|outpatient|inpatient"
        }
        for qt, pat in patterns.items():
            if re.search(pat, q):
                return qt
        return "general"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query for QuantAI Hospital context."""
        # Remove common stopwords and punctuation
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about',
            'show', 'tell', 'give', 'find', 'get', 'me', 'please', 'list', 'details', 'info', 'information',
            'hospital', 'quantai', 'auckland', 'nz', 'new', 'zealand'
        }
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]

        # Add specific hospital ID patterns
        id_patterns = [
            (r'PT\d{6}', 'patient_id'),
            (r'AP\d{6}', 'appointment_id'),
            (r'ST\d{6}', 'staff_id'),
            (r'EQ\d{6}', 'equipment_id'),
            (r'IV\d{6}', 'inventory_id'),
            (r'IN\d{4}', 'infrastructure_id')
        ]
        for pattern, _ in id_patterns:
            ids = re.findall(pattern, query.upper())
            keywords.extend(ids)

        # Add Auckland-specific terms if mentioned
        auckland_terms = [
            "cbd", "central", "north shore", "south", "west", "east",
            "waitakere", "manukau", "waitematā", "dhb", "district"
        ]
        for term in auckland_terms:
            if term in query.lower():
                keywords.append(term)

        return list(set(keywords))

    def _determine_response_type(self, query: str) -> str:
        """Determine the appropriate response type based on query content."""
        q = query.lower()
        if any(x in q for x in ["urgent", "critical", "emergency", "stat", "immediate", "now"]):
            return "operational"
        if any(x in q for x in ["trend", "rate", "compare", "statistics", "analysis", "analytics", "report"]):
            return "analytical"
        return "standard"

    def _format_response(self, resp: str, q_type: str) -> str:
        """Format the response with explicit QuantAI Hospital (Auckland) branding and context."""
        resp = resp.strip()
        for pat in [r"As an AI", r"I'm an AI", r"artificial intelligence"]:
            resp = re.sub(pat, "QuantAI Hospital Assistant", resp, flags=re.IGNORECASE)

        prefixes = {
            "patient": "QuantAI Hospital Auckland | Patient Information",
            "appointment": "QuantAI Hospital Auckland | Appointment Details",
            "staff": "QuantAI Hospital Auckland | Staff Information",
            "equipment": "QuantAI Hospital Auckland | Equipment Status",
            "inventory": "QuantAI Hospital Auckland | Inventory Status",
            "infrastructure": "QuantAI Hospital Auckland | Infrastructure Information",
            "medical": "QuantAI Hospital Auckland | Medical Information",
            "history": "QuantAI Hospital Auckland | Medical History",
            "schedule": "QuantAI Hospital Auckland | Staff Schedule",
            "visit": "QuantAI Hospital Auckland | Visitor Information"
        }
        heading = prefixes.get(q_type, "QuantAI Hospital Auckland | Information")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M NZST")
        return f"{heading} | {timestamp}\n\n{resp}"

    async def process_query(self, query: str) -> str:
        """
        Process a query and generate a strictly NZ hospital-specific response as QuantAI Hospital Assistant.
        """
        try:
            q_type = self._determine_query_type(query)
            keywords = self._extract_keywords(query)
            resp_type = self._determine_response_type(query)

            relevant = self.data_loader.get_relevant_data(q_type, keywords)
            context_parts: List[str] = [f"Information Type (NZ): {q_type}"]

            for src, rows in relevant.items():
                context_parts.append(f"\n{src.replace('_', ' ').title()}:")
                for r in rows:
                    row_str = ", ".join([f"{k}: {v}" for k, v in r.items() if k.lower() in {"id", "name", "type", "status", "date", "department", "patient_id", "appointment_id"}])
                    context_parts.append(row_str)

            context = "\n".join(context_parts)
            llm_resp = await self.gemma_client.generate_response(query, context, "", resp_type)
            return self._format_response(llm_resp, q_type)
        except Exception as e:
            logger.error(f"HospitalRAG error: {e}")
            return "System Error: Unable to process query."

    async def close(self):
        if self.gemma_client:
            await self.gemma_client.close()

# Initialize the RAG system
# Obsolete initialisation commented out
# rag_system = AustralianPoliceRAG()

# ======================  QuantAI Hospital Assistant ======================

class HospitalDataLoader:
    """Loads and manages QuantAI Hospital datasets with advanced filtering capabilities."""

    def __init__(self):
        self.data_dir = "data"
        self.data: Dict[str, Any] = {}
        self.load_data()

    def load_data(self):
        """Load infrastructure JSON and all CSV files prefixed with 'quantai_hospital_'"""
        try:
            infra_path = os.path.join(self.data_dir, "hospital_infrastructure.json")
            if os.path.exists(infra_path):
                with open(infra_path, "r") as f:
                    self.data["infrastructure"] = json.load(f)
            else:
                logger.warning("hospital_infrastructure.json not found in data directory")

            csv_files = [f for f in os.listdir(self.data_dir) if f.startswith("quantai_hospital_") and f.endswith(".csv")]

            for file in csv_files:
                dataset_name = file.replace("quantai_hospital_", "").replace(".csv", "")
                path = os.path.join(self.data_dir, file)
                try:
                    self.data[dataset_name] = pd.read_csv(path)
                    logger.info(f"Loaded {file}")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")

            if not self.data:
                logger.warning("No QuantAI hospital datasets loaded – verify /data directory")
        except Exception as e:
            logger.error(f"HospitalDataLoader error: {e}")
            raise

    def get_relevant_data(self, query_type: str, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve rows most relevant to the query type and keywords."""
        try:
            mapping = {
                "patient": ["patients", "medical_history"],
                "appointment": ["appointments", "patients", "staff_schedule"],
                "staff": ["staff_schedule"],
                "equipment": ["equipment_maintenance", "inventory_management"],
                "inventory": ["inventory_management"],
                "department": ["infrastructure", "staff_schedule"],
                "medical": ["medical_history"],
                "infrastructure": ["infrastructure"],
                "general": ["infrastructure"],
            }

            relevant: Dict[str, Any] = {}
            sources = mapping.get(query_type, ["infrastructure"])

            for src in sources:
                if src not in self.data:
                    continue

                df = self.data[src]
                filtered = df

                if keywords:
                    mask_total = None
                    for kw in keywords:
                        mask_kw = None
                        for col in df.columns:
                            m = df[col].astype(str).str.contains(kw, case=False, na=False, regex=True)
                            mask_kw = m if mask_kw is None else mask_kw | m
                        mask_total = mask_kw if mask_total is None else mask_total & mask_kw
                    if mask_total is not None:
                        filtered = filtered[mask_total]

                date_cols = [c for c in filtered.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols:
                    try:
                        filtered = filtered.sort_values(by=date_cols[0], ascending=False)
                    except Exception:
                        pass

                if not filtered.empty:
                    relevant[src] = filtered.head(3).to_dict("records")

            return relevant
        except Exception as e:
            logger.error(f"get_relevant_data error: {e}")
            return {}

class HospitalRAG:
    """RAG system powering *Hospital Assistant by QuantAI, NZ* with follow-up and context awareness."""

    def __init__(self):
        self._load_env()
        self._init_components()

    def _load_env(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY missing from environment")

    def _init_components(self):
        self.data_loader = HospitalDataLoader()
        self.gemma_client = NZOpenRouterClient(self.api_key)

    # ---------- Query Understanding ----------
    def _determine_query_type(self, query: str) -> str:
        q = query.lower()
        patterns = {
            "patient": r"patient|demographic|medical history|diagnosis|condition",
            "appointment": r"appointment|schedule|booking|visit",
            "staff": r"staff|doctor|nurse|surgeon|technician|physician|employee",
            "equipment": r"equipment|device|machine|maintenance",
            "inventory": r"inventory|stock|supply|pharmacy",
            "department": r"department|ward|icu|unit|clinic|radiology|cardiology|pediatrics",
            "medical": r"medication|treatment|therapy|lab|test|result",
            "emergency": r"emergency|urgent|critical|triage|code",
        }
        for qt, pat in patterns.items():
            if re.search(pat, q):
                return qt
        return "general"

    def _extract_keywords(self, query: str) -> list:
        stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "about"}
        tokens = re.findall(r"\b\w+\b", query.lower())
        return [t for t in tokens if t not in stop and len(t) > 2]

    def _determine_response_type(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["urgent", "critical", "emergency", "stat"]):
            return "operational"
        if any(x in q for x in ["trend", "rate", "compare", "statistics", "analysis", "analytics"]):
            return "analytical"
        return "standard"

    def _format_response(self, resp: str, q_type: str) -> str:
        resp = resp.strip()
        for pat in [r"As an AI", r"I'm an AI", r"artificial intelligence"]:
            resp = re.sub(pat, "The assistant", resp, flags=re.IGNORECASE)
        prefixes = {
            "patient": "Patient Info",
            "appointment": "Appointment",
            "staff": "Staff Info",
            "equipment": "Equipment",
            "inventory": "Inventory",
            "department": "Department",
            "medical": "Medical",
            "emergency": "Emergency",
        }
        heading = prefixes.get(q_type, "Info")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{heading} | {timestamp}\n{resp}"

    # ---------- Public API ----------
    async def process_query(self, query: str, conversation_context: str = "") -> str:
        try:
            q_type = self._determine_query_type(query)
            keywords = self._extract_keywords(query)
            resp_type = self._determine_response_type(query)
            relevant = self.data_loader.get_relevant_data(q_type, keywords)
            context_parts = [f"Type: {q_type}"]
            for src, rows in relevant.items():
                context_parts.append(f"\n{src.replace('_', ' ').title()}:")
                for r in rows:
                    row_str = ", ".join([f"{k}: {v}" for k, v in r.items() if k.lower() in {"id", "name", "type", "status", "date", "department", "patient_id", "appointment_id"}])
                    context_parts.append(row_str)
            context = "\n".join(context_parts)
            llm_resp = await self.gemma_client.generate_response(query, context, conversation_context, resp_type)
            return self._format_response(llm_resp, q_type)
        except Exception as e:
            logger.error(f"HospitalRAG error: {e}")
            return "Sorry, I couldn't get that info. Could you try rephrasing?"

    async def close(self):
        if self.gemma_client:
            await self.gemma_client.close()

# ======================  End Hospital Assistant ======================

# Initialize the Hospital RAG system
rag_system = HospitalRAG() 