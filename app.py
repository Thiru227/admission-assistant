"""
RTC Scholar - Production RAG Agent (Clean Version)
==================================================
Pure RAG functionality - Frontend handles lead collection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import re
from typing import List, Tuple
from collections import Counter
import time
import math

from knowledge_base import KNOWLEDGE_BASE

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# PRODUCTION VECTOR DB WITH BM25
# ============================================
class ProductionVectorDB:
    """Production-grade retrieval with BM25 ranking"""
    
    def __init__(self):
        self.documents = []
        self.doc_frequencies = {}
        self.avg_doc_length = 0
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75  # BM25 parameter
        print(f"✓ ProductionVectorDB initialized with BM25 ranking")
    
    def add_documents(self, docs: List[str]):
        """Add documents and build inverted index"""
        self.documents.extend(docs)
        self._build_index()
        print(f"✓ Indexed {len(docs)} documents with BM25")
    
    def _build_index(self):
        """Build inverted index and calculate document frequencies"""
        for doc in self.documents:
            terms = set(self.extract_keywords(doc))
            for term in terms:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
        
        total_length = sum(len(self.extract_keywords(doc)) for doc in self.documents)
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        
        print(f"✓ Index built: {len(self.doc_frequencies)} unique terms")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords with stop word removal"""
        stop_words = {
            'what', 'is', 'the', 'who', 'where', 'when', 'how', 'are', 'do',
            'does', 'about', 'tell', 'me', 'can', 'you', 'a', 'an', 'and',
            'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'that', 'this', 'these', 'those', 'be', 'been', 'being', 'have',
            'has', 'had', 'was', 'were', 'will', 'would', 'could', 'should'
        }
        
        normalized = self.normalize_text(text)
        words = [w for w in normalized.split() if w not in stop_words and len(w) > 2]
        return words
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        query_lower = query.lower()
        expansions = []
        
        # Role expansions
        role_map = {
            'principal': ['principal', 'head', 'director'],
            'ceo': ['ceo', 'chief executive', 'executive officer'],
            'chairman': ['chairman', 'chairperson', 'chair'],
            'vice principal': ['vice principal', 'deputy principal'],
            'dean': ['dean', 'head of research', 'research head'],
            'placement': ['placement', 'career', 'jobs', 'recruitment'],
            'training': ['training', 'career development', 'skill development']
        }
        
        for role, synonyms in role_map.items():
            if role in query_lower:
                expansions.extend(synonyms)
        
        # Department expansions
        dept_map = {
            'cse': ['computer science', 'cse', 'cs'],
            'ece': ['electronics', 'ece', 'communication'],
            'eee': ['electrical', 'eee', 'electronics'],
            'mechanical': ['mechanical', 'mech', 'me'],
            'civil': ['civil', 'ce']
        }
        
        for dept, synonyms in dept_map.items():
            if dept in query_lower:
                expansions.extend(synonyms)
        
        return list(set(expansions))
    
    def calculate_idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency"""
        doc_freq = self.doc_frequencies.get(term, 0)
        if doc_freq == 0:
            return 0
        return math.log((len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def calculate_bm25_score(self, query_terms: List[str], doc: str) -> float:
        """Calculate BM25 relevance score"""
        doc_terms = self.extract_keywords(doc)
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0
        
        doc_term_freq = Counter(doc_terms)
        score = 0
        
        for term in query_terms:
            if term not in doc_term_freq:
                continue
            
            tf = doc_term_freq[term]
            idf = self.calculate_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def calculate_phrase_bonus(self, query: str, doc: str) -> float:
        """Give bonus for exact phrase matches"""
        query_normalized = self.normalize_text(query)
        doc_normalized = self.normalize_text(doc)
        
        if query_normalized in doc_normalized:
            position = doc_normalized.find(query_normalized)
            position_score = max(0, 50 - position / 10)
            return 100 + position_score
        
        return 0
    
    def calculate_entity_bonus(self, query: str, doc: str) -> float:
        """Bonus for named entity matches"""
        query_lower = query.lower()
        doc_lower = doc.lower()
        
        entities = [
            'nagaraj', 'balakrishnan', 'manickam', 'sendhil', 'madan',
            'geetha', 'senthilkumar', 'saravanan', 'priya', 'ramachandran',
            'coimbatore', 'rathinam', 'rtc', 'aicte', 'naac', 'nirf',
            'btech', 'mtech', 'phd', 'mba'
        ]
        
        score = 0
        for entity in entities:
            if entity in query_lower and entity in doc_lower:
                score += 30
        
        return score
    
    def search(self, query: str, top_k: int = 4) -> List[str]:
        """Advanced search with multiple ranking signals"""
        if not query or not self.documents:
            return []
        
        query_terms = self.extract_keywords(query)
        expanded_terms = self.expand_query(query)
        all_terms = list(set(query_terms + expanded_terms))
        
        scored_docs = []
        
        for doc in self.documents:
            bm25_score = self.calculate_bm25_score(all_terms, doc)
            phrase_bonus = self.calculate_phrase_bonus(query, doc)
            entity_bonus = self.calculate_entity_bonus(query, doc)
            
            final_score = (bm25_score * 1.0) + phrase_bonus + entity_bonus
            
            if final_score > 0:
                scored_docs.append((final_score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        if scored_docs:
            print(f"🔍 Query: '{query}' → Top score: {scored_docs[0][0]:.2f}")
        
        return [doc for _, doc in scored_docs[:top_k]]

# ============================================
# INITIALIZE VECTOR DB
# ============================================
vector_db = ProductionVectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)
print(f"📚 Knowledge base ready: {len(KNOWLEDGE_BASE)} documents")

# ============================================
# OPENROUTER LLM CONFIGURATION
# ============================================
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

def is_admission_related(query: str) -> bool:
    """Check if query is admission-related for special handling"""
    keywords = [
        'admission', 'apply', 'application', 'join', 'enroll', 'course',
        'program', 'eligibility', 'fees', 'fee structure', 'how to join',
        'registration', 'seat', 'interested', 'want to study'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

def call_llm(prompt: str, context: str) -> str:
    """Call OpenRouter API with RAG context"""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ Configuration needed. Please set OPENROUTER_API_KEY."
    
    # Check if query is admission-related
    is_admission = is_admission_related(prompt)
    
    if is_admission:
        # Special prompt for admission queries - naturally lead to brochure offer
        system_prompt = f"""You are RTC Scholar, a friendly and enthusiastic admission counselor for Rathinam Technical Campus.

CONTEXT:
{context}

YOUR TASK:
1. Answer their admission question clearly using the context
2. Be excited and positive about RTC
3. After answering, NATURALLY suggest: "Would you like me to send you our detailed brochure with complete admission information? Just click the 'Get Brochure' button above! 📚"
4. Keep response under 120 words
5. Sound conversational and helpful

PERSONALITY: Warm, encouraging, student-friendly"""
    else:
        # Normal prompt for general queries
        system_prompt = f"""You are RTC Scholar, a helpful AI assistant for Rathinam Technical Campus.

CONTEXT:
{context}

RULES:
1. Answer using ONLY the context provided
2. Be accurate and concise
3. Keep response under 100 words
4. End with: "Anything else you'd like to know? 😊"

PERSONALITY: Professional, friendly, helpful"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "RTC-Scholar"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    
    except requests.exceptions.Timeout:
        return "Sorry, response took too long. Please try again! ⏱️"
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM Error: {e}")
        return "Having trouble connecting. Please try again! 🔄"
    except (KeyError, IndexError) as e:
        print(f"❌ Parse Error: {e}")
        return "Something went wrong. Please try again! 😅"

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'RTC Scholar - Production RAG Agent',
        'version': '3.0-clean',
        'features': ['BM25 ranking', 'Query expansion', 'Phrase matching'],
        'documents': len(vector_db.documents),
        'api_configured': bool(OPENROUTER_API_KEY)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'documents': len(vector_db.documents),
        'api_key': bool(OPENROUTER_API_KEY)
    }), 200

@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """Main webhook - Pure RAG, no lead logic"""
    
    try:
        req = request.get_json(silent=True, force=True)
        query_text = req.get('queryResult', {}).get('queryText', '')
        
        if not query_text:
            return jsonify({
                'fulfillmentText': 'Sorry, I didn\'t get that. Could you ask again?'
            })
        
        print(f"📥 Query: {query_text}")
        
        # Retrieve relevant documents
        relevant_docs = vector_db.search(query_text, top_k=4)
        
        if not relevant_docs:
            return jsonify({
                'fulfillmentText': 'Hmm, I couldn\'t find specific info about that. Could you rephrase? 🤔'
            })
        
        context = "\n\n".join(relevant_docs)
        print(f"✓ Retrieved {len(relevant_docs)} docs")
        
        # Generate response
        response_text = call_llm(query_text, context)
        print(f"✓ Response: {response_text[:80]}...")
        
        return jsonify({
            'fulfillmentText': response_text,
            'source': 'webhook'
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'fulfillmentText': 'Oops! Something went wrong. Try again! 🔄'
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint with debugging"""
    
    data = request.get_json()
    query = data.get('query', 'Who is the principal?')
    
    relevant_docs = vector_db.search(query, top_k=3)
    context = "\n\n".join(relevant_docs) if relevant_docs else "No context"
    response = call_llm(query, context)
    
    return jsonify({
        'query': query,
        'documents_found': len(relevant_docs),
        'retrieved_docs': relevant_docs,
        'response': response,
        'is_admission_query': is_admission_related(query)
    })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List sample documents"""
    return jsonify({
        'total': len(vector_db.documents),
        'sample': vector_db.documents[:5]
    })

# ============================================
# RUN SERVER
# ============================================
# ============================================
# RUN SERVER (Render-compatible)
# ============================================
def start():
    """Start Flask app for Render"""
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 RTC Scholar (Clean RAG) starting on port {port}")
    print(f"📊 Documents: {len(vector_db.documents)}")
    print(f"🔑 API Key configured: {bool(OPENROUTER_API_KEY)}")
    print(f"📝 Lead collection handled on frontend")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    start()

