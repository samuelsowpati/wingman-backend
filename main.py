"""
Main FastAPI Application

This is the central API server for the Air Force Q&A chatbot.
It processes local PDF files, searches documents, and answers questions 
about Air Force policies, procedures, and general information.

Key Concepts:
- FastAPI: Modern Python web framework for building APIs
- ASGI: Asynchronous Server Gateway Interface (handles concurrent requests)  
- Pydantic: Data validation and serialization
- CORS: Cross-Origin Resource Sharing (allows React frontend to connect)
- RESTful API: Standard web API design patterns
- RAG: Retrieval-Augmented Generation for accurate Q&A
- Local Processing: Works with local PDF files only
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import uvicorn
import os
import logging
import re
import asyncio
import time
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Import our custom modules (after environment is loaded)
from embeddings import embedding_service
from pdf_processor import AFH1PDFProcessor

# Initialize the PDF processor with default settings
pdf_processor = AFH1PDFProcessor()
from vector_store import vector_store
from llm_service import llm_service


class QueryProcessor:
    """Enhanced query processing for Air Force Q&A."""
    
    @staticmethod
    def expand_air_force_query(query: str) -> List[str]:
        """Expand Air Force queries with related terms for better semantic matching."""
        expanded_queries = [query]
        query_lower = query.lower()
        
                    # Air Force terminology and abbreviations
        af_expansions = {
            # Organizational units
            'usaf': ['united states air force', 'air force', 'us air force'],
            'dod': ['department of defense', 'defense department'],
            'hq': ['headquarters'],
            'aetc': ['air education and training command', 'training command'],
            'acc': ['air combat command', 'combat command'],
            'amc': ['air mobility command', 'mobility command'],
            'afmc': ['air force materiel command', 'materiel command'],
            'afsoc': ['air force special operations command', 'special operations'],
            'pacaf': ['pacific air forces'],
            'usafe': ['united states air forces in europe'],
            
            # Aviation and combat terms
            'flying aces': ['aviation history', 'combat pilots', 'ace pilots', 'fighter pilots'],
            'aces': ['aviation history', 'combat pilots', 'ace pilots', 'fighter pilots'],
            'fighter pilots': ['aviation history', 'combat pilots', 'military aviation'],
            'combat aviation': ['aviation history', 'military aviation'],
            
            # Personnel and roles
            'airman': ['air force member', 'service member', 'military member'],
            'nco': ['non-commissioned officer', 'noncom'],
            'snco': ['senior non-commissioned officer', 'senior noncom'],
            'officer': ['commissioned officer'],
            'enlisted': ['enlisted member', 'enlisted personnel'],
            
            # Concepts and procedures
            'pcs': ['permanent change of station', 'move', 'relocation'],
            'tdy': ['temporary duty', 'temporary assignment'],
            'pt': ['physical training', 'fitness', 'physical fitness'],
            'epr': ['enlisted performance report', 'performance evaluation'],
            'opr': ['officer performance report', 'officer evaluation'],
            'decoration': ['medal', 'award', 'recognition'],
            'promotion': ['advancement', 'rank increase'],
            'deployment': ['overseas assignment', 'combat tour'],
            
            # Training and education
            'bmt': ['basic military training', 'boot camp', 'basic training'],
            'tech school': ['technical training', 'job training'],
            'ait': ['advanced individual training'],
            'pme': ['professional military education'],
            'ncoa': ['noncommissioned officer academy'],
            'als': ['airman leadership school'],
            
            # Benefits and services
            'bah': ['basic allowance for housing', 'housing allowance'],
            'bas': ['basic allowance for subsistence', 'food allowance'],
            'tricare': ['military health insurance', 'medical coverage'],
            'commissary': ['military grocery store'],
            'bx': ['base exchange', 'military store'],
            'mwr': ['morale welfare recreation', 'recreational services'],
        }
        
        # Add expanded versions for known terms
        for abbr, expansions in af_expansions.items():
            if abbr in query_lower:
                for expansion in expansions:
                    expanded_queries.append(query_lower.replace(abbr, expansion))
        
        # Topic-based query expansion
        topic_keywords = {
            'leadership': ['command', 'management', 'supervision', 'guidance'],
            'training': ['education', 'instruction', 'learning', 'development'],
            'career': ['advancement', 'progression', 'development', 'growth'],
            'deployment': ['overseas', 'combat', 'mission', 'assignment'],
            'uniform': ['dress', 'appearance', 'clothing', 'attire'],
            'fitness': ['physical training', 'pt', 'health', 'wellness'],
            'family': ['dependents', 'spouse', 'children', 'relatives'],
            'housing': ['quarters', 'dormitory', 'residence', 'living'],
            'pay': ['salary', 'compensation', 'allowance', 'benefits'],
            'discipline': ['punishment', 'corrective action', 'misconduct'],
        }
        
        # Add topic-related terms
        for topic, related_terms in topic_keywords.items():
            if topic in query_lower:
                for term in related_terms:
                    if term not in query_lower:
                        expanded_queries.append(f"{query} {term}")
        
        # Remove duplicates and return
        return list(set(expanded_queries))[:5]  # Limit to 5 queries to avoid overwhelming


def merge_search_results(search_results: List[Dict]) -> List[Dict]:
    """Merge and rank search results from multiple queries."""
    
    # Group by document chunk to avoid duplicates
    chunk_scores = {}
    
    for result in search_results:
        chunk_id = result.get('chunk_id', '')
        source = result.get('source', '')
        key = f"{source}_{chunk_id}"
        
        if key not in chunk_scores:
            chunk_scores[key] = {
                'result': result,
                'max_score': result['score'],
                'query_count': 1,
                'query_indices': [result.get('query_index', 0)]
            }
        else:
            # Take the highest score and increment query count
            chunk_scores[key]['max_score'] = max(
                chunk_scores[key]['max_score'], 
                result['score']
            )
            chunk_scores[key]['query_count'] += 1
            chunk_scores[key]['query_indices'].append(result.get('query_index', 0))
    
    # Calculate boosted scores based on multiple query matches
    for key, data in chunk_scores.items():
        # Boost score if multiple queries matched this chunk
        query_boost = min(data['query_count'] * 0.05, 0.15)  # Max 15% boost
        data['boosted_score'] = data['max_score'] + query_boost
    
    # Sort by boosted score and return top results
    merged_results = [
        data['result'] for data in sorted(
            chunk_scores.values(), 
            key=lambda x: x['boosted_score'], 
            reverse=True
        )
    ]
    
    return merged_results[:10]  # Return top 10 merged results


# ================================
# Enhanced Query Processing
# ================================

class EnhancedQueryProcessor:
    """Advanced query processing for Air Force Q&A"""
    
    def __init__(self):
        self.af_terminology = {
            # Aircraft and Aviation
            'flying aces': ['ace pilots', 'combat pilots', 'fighter aces', 'aviation heroes', 'top pilots'],
            'pilot': ['aviator', 'aircrew', 'flight crew', 'navigator'],
            
            # Organizations and Commands
            'usaf': ['united states air force', 'air force', 'us air force'],
            'aetc': ['air education and training command', 'training command'],
            'acc': ['air combat command', 'combat command'],
            'amc': ['air mobility command', 'mobility command'],
            
            # Military Concepts
            'mission': ['purpose', 'objective', 'goal', 'mandate'],
            'responsibility': ['duty', 'obligation', 'role', 'function'],
            'leadership': ['command', 'management', 'supervision', 'guidance'],
            'training': ['education', 'instruction', 'learning', 'development'],
        }
        
        self.query_patterns = {
            'definition': [r'what (is|are)', r'define', r'meaning of'],
            'procedure': [r'how (do|does|to)', r'what is the process', r'steps to'],
            'responsibility': [r'who is responsible', r'responsibilities of', r'duties of'],
            'history': [r'history of', r'when did', r'origin of']
        }
        
        self.stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Comprehensive query analysis"""
        cleaned_query = self._clean_query(query)
        query_type = self._classify_query(cleaned_query)
        key_terms = self._extract_key_terms(cleaned_query)
        expanded_queries = self._expand_query(cleaned_query, key_terms, query_type)
        confidence = self._calculate_confidence(cleaned_query, key_terms)
        
        return {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'query_type': query_type,
            'expanded_queries': expanded_queries,
            'key_terms': key_terms,
            'confidence': confidence
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[.!]+$', '?', query)
        return query
    
    def _classify_query(self, query: str) -> str:
        """Classify the query type"""
        query_lower = query.lower()
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        return 'general'
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Add multi-word phrases
        for term in self.af_terminology.keys():
            if term in query.lower():
                key_terms.append(term)
        
        return list(set(key_terms))
    
    def _expand_query(self, query: str, key_terms: List[str], query_type: str) -> List[str]:
        """Expand query with related terms and synonyms"""
        expanded_queries = [query]
        query_lower = query.lower()
        
        # Expand based on Air Force terminology
        for term in key_terms:
            if term in self.af_terminology:
                for variant in self.af_terminology[term][:2]:  # Limit to 2 variants
                    expanded_query = query_lower.replace(term, variant)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
        
        return list(dict.fromkeys(expanded_queries))[:5]
    
    def _calculate_confidence(self, query: str, key_terms: List[str]) -> float:
        """Calculate confidence score for query understanding"""
        confidence = 0.5
        
        # Boost for recognized Air Force terms
        af_terms_found = sum(1 for term in key_terms if term in self.af_terminology)
        if af_terms_found > 0:
            confidence += min(af_terms_found * 0.1, 0.3)
        
        # Boost for complete questions
        if query.endswith('?'):
            confidence += 0.1
        
        # Penalize for very short queries
        if len(query.split()) < 3:
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)

# Global instances
enhanced_query_processor = EnhancedQueryProcessor()


# ================================
# Enhanced Models and Enums
# ================================

class ConfidenceLevel(str, Enum):
    """Confidence levels for answers"""
    HIGH = "high"        # Score > 0.8
    MEDIUM = "medium"    # Score 0.5-0.8  
    LOW = "low"         # Score < 0.5


# ================================
# Enhanced Answer Quality Functions
# ================================

def enhance_llm_prompt(user_question: str, context_chunks: List[Dict[str, Any]], query_analysis: Dict) -> str:
    """Create an enhanced prompt for better LLM responses"""
    context_text = build_enhanced_context(context_chunks)
    answer_style = get_answer_style(query_analysis.get('query_type', 'general'))
    
    prompt = f"""You are an expert Air Force knowledge assistant. Answer the user's question based ONLY on the provided Air Force documentation.

QUESTION: {user_question}

QUERY TYPE: {query_analysis.get('query_type', 'general')}
ANSWER STYLE: {answer_style}

CONTEXT FROM AIR FORCE DOCUMENTS:
{context_text}

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Be specific and detailed
3. Use exact terminology from the documents
4. Include relevant numbers, dates, and specific references when available
5. If the context doesn't fully answer the question, state what information is available
6. Use authoritative language (e.g., "According to AFH1...", "The regulation states...")
7. Structure your answer clearly with main points
8. Do not make assumptions or add information not in the context

ANSWER:"""
    
    return prompt

def build_enhanced_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Build enhanced context from chunks with better formatting"""
    if not context_chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')
        
        # Clean content
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        cleaned_content = re.sub(r' +', ' ', cleaned_content)
        
        source_info = f"[Source {i}: {metadata.get('doc_type', 'Unknown')} - {metadata.get('source', 'Unknown')}]"
        context_parts.append(f"{source_info}\n{cleaned_content}")
    
    return "\n\n" + "\n\n---\n\n".join(context_parts)

def get_answer_style(query_type: str) -> str:
    """Get appropriate answer style based on query type"""
    styles = {
        'definition': 'Provide a clear definition followed by detailed explanation',
        'procedure': 'List step-by-step procedures in order',
        'responsibility': 'Clearly state roles and responsibilities with specific duties',
        'history': 'Provide chronological historical information with dates and context',
        'general': 'Provide comprehensive information addressing all aspects of the question'
    }
    return styles.get(query_type, styles['general'])

def calculate_dynamic_threshold(query_analysis: Dict, search_results: List[Dict]) -> float:
    """Calculate dynamic relevance threshold based on query and results"""
    base_threshold = 0.4
    
    # Adjust based on query confidence
    confidence = query_analysis.get('confidence', 0.5)
    if confidence >= 0.8:
        base_threshold += 0.1
    elif confidence <= 0.5:
        base_threshold -= 0.1
    
    # Adjust based on query type
    query_type = query_analysis.get('query_type', 'general')
    type_adjustments = {
        'definition': 0.05,
        'history': -0.05,
        'general': -0.05
    }
    base_threshold += type_adjustments.get(query_type, 0)
    
    return max(min(base_threshold, 0.7), 0.2)

def determine_confidence_level(score: float) -> ConfidenceLevel:
    """Determine confidence level based on score"""
    if score >= 0.8:
        return ConfidenceLevel.HIGH
    elif score >= 0.6:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW

def assess_answer_quality(answer: str, relevance_scores: List[float]) -> Dict[str, Any]:
    """Assess the quality of a generated answer"""
    # Base confidence from relevance
    base_confidence = max(relevance_scores) if relevance_scores else 0.0
    
    # Adjust based on answer characteristics
    word_count = len(answer.split())
    if word_count >= 50:
        base_confidence += 0.1
    elif word_count < 20:
        base_confidence -= 0.2
    
    # Check for specific details
    specific_indicators = len(re.findall(r'\b\d{4}\b|\d+\.\d+|AFH\s*\d+|AFI\s*\d+', answer))
    base_confidence += min(specific_indicators * 0.05, 0.15)
    
    # Check for authoritative language
    auth_patterns = [r'according to', r'as stated in', r'mandates that', r'requires that']
    auth_count = sum(1 for pattern in auth_patterns if re.search(pattern, answer, re.IGNORECASE))
    base_confidence += min(auth_count * 0.1, 0.2)
    
    confidence_score = min(max(base_confidence, 0.0), 1.0)
    
    return {
        'confidence_score': confidence_score,
        'word_count': word_count,
        'has_specific_details': specific_indicators > 0,
        'has_authoritative_language': auth_count > 0
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Air Force Q&A API",
    description="API for Air Force Question & Answer chatbot using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc at http://localhost:8000/redoc
)

# Configure CORS for React frontend
# This allows your React app to call this API from a different port
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ================================
# Pydantic Models (Data Validation)
# ================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's question or message", min_length=1)
    history: List[Dict[str, str]] = Field(default=[], description="Previous chat history")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What are the responsibilities of SAF/AQ?",
                "history": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI-generated response")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used")
    processing_time: Optional[float] = Field(None, description="Time taken to process (seconds)")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "The Assistant Secretary of the Air Force for Acquisition (SAF/AQ) maintains responsibility for...",
                "sources": [
                    {
                        "document": "afi10-2402.pdf",
                        "score": 0.89,
                        "doc_type": "AFI"
                    }
                ],
                "processing_time": 1.23
            }
        }


# ================================
# Enhanced Q&A Models (No Chat History)
# ================================

class QuestionRequest(BaseModel):
    """Enhanced request model for Q&A."""
    question: str = Field(..., description="User's question", min_length=1, max_length=500)
    include_sources: bool = Field(default=True, description="Include source citations")
    max_sources: int = Field(default=3, description="Maximum number of sources to return", ge=1, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are flying aces?",
                "include_sources": True,
                "max_sources": 3
            }
        }

class SourceCitation(BaseModel):
    """Source citation with enhanced metadata"""
    document: str = Field(..., description="Source document name")
    doc_type: str = Field(..., description="Document type (AFH1, AFI, etc.)")
    chunk_id: int = Field(..., description="Chunk identifier")
    chapter_number: Optional[int] = Field(None, description="Chapter number")
    chapter_title: Optional[str] = Field(None, description="Chapter title")
    section_id: Optional[str] = Field(None, description="Section identifier")
    section_title: Optional[str] = Field(None, description="Section title")
    start_page: Optional[int] = Field(None, description="Starting page number")
    end_page: Optional[int] = Field(None, description="Ending page number")
    edition: Optional[str] = Field(None, description="Document edition/date")
    granularity: str = Field("section_chunk", description="Chunk granularity (section_chunk or micro_chunk)")
    relevance_score: float = Field(..., description="Similarity score (0-1)")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    preview: str = Field(..., description="Text preview (first 100 chars)")

class AnswerResponse(BaseModel):
    """Enhanced response model for Q&A."""
    answer: str = Field(..., description="AI-generated answer")
    confidence: ConfidenceLevel = Field(..., description="Overall answer confidence")
    relevance_score: float = Field(..., description="Best relevance score from sources")
    sources: List[SourceCitation] = Field(default=[], description="Source citations")
    processing_time: float = Field(..., description="Processing time in seconds")
    chunks_found: int = Field(..., description="Total relevant chunks found")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "Flying aces are military pilots who have achieved five or more confirmed aerial combat victories...",
                "confidence": "high",
                "relevance_score": 0.89,
                "sources": [
                    {
                        "document": "afh1.pdf",
                        "doc_type": "AFH1", 
                        "chunk_id": 42,
                        "relevance_score": 0.89,
                        "confidence": "high",
                        "preview": "Flying aces represent the pinnacle of aerial combat excellence..."
                    }
                ],
                "processing_time": 1.23,
                "chunks_found": 5
            }
        }


class ProcessPDFRequest(BaseModel):
    """Request model for PDF processing."""
    pdf_file: str = Field(..., description="Local PDF file path to process")
    
    class Config:
        schema_extra = {
            "example": {
                "pdf_file": "afh1.pdf"
            }
        }


class ProcessPDFResponse(BaseModel):
    """Response model for PDF processing."""
    total_files: int = Field(..., description="Number of PDF files processed")
    total_chunks: int = Field(..., description="Total document chunks created")
    successful: int = Field(..., description="Successfully processed PDFs")
    failed: int = Field(..., description="Failed PDF processing attempts")
    processing_time: float = Field(..., description="Total processing time (seconds)")


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    doc_types: Optional[List[str]] = Field(default=None, description="Filter by document types")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "acquisition responsibilities",
                "top_k": 5,
                "doc_types": ["AFI", "AFMAN"]
            }
        }


class SearchResponse(BaseModel):
    """Response model for document search."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")


# ================================
# API Endpoints
# ================================

@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    
    This is the simplest endpoint that confirms the API is running.
    """
    return {
        "message": "🚀 Air Force Q&A API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy",
        "description": "Ask any question about Air Force policies, procedures, and information"
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.
    
    This endpoint checks if all services are working properly.
    """
    try:
        # Check embedding service
        embedding_info = embedding_service.get_model_info()
        
        # Check vector store
        vector_stats = await vector_store.get_index_stats()
        
        # Check PDF processor
        processor_info = pdf_processor.get_processor_stats()
        
        return {
            "status": "healthy",
            "services": {
                "embedding_service": {
                    "status": "ok",
                    "model": embedding_info["model_name"],
                    "dimension": embedding_info["dimension"]
                },
                "vector_store": {
                    "status": "ok",
                    "total_documents": vector_stats.get("total_vectors", 0),
                    "index_name": vector_store.index_name
                },
                "pdf_processor": {
                    "status": "ok",
                    "chunk_size": processor_info["chunk_size"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Enhanced Q&A endpoint for Air Force questions.
    
    This endpoint provides:
    - Advanced query understanding and expansion
    - Confidence scoring for answers
    - Quality assessment of responses
    - Enhanced source citations
    - Better error handling
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Processing question: {request.question[:100]}...")
        
        # Step 1: Enhanced query analysis
        query_analysis = enhanced_query_processor.analyze_query(request.question)
        logger.info(f"📊 Query analysis - Type: {query_analysis['query_type']}, Confidence: {query_analysis['confidence']:.2f}")
        
        # Step 2: Get embeddings for expanded queries
        query_embeddings = await asyncio.gather(*[
            embedding_service.get_embedding(query) for query in query_analysis['expanded_queries']
        ])
        
        # Step 3: Enhanced search with multiple queries
        all_search_results = []
        for i, embedding in enumerate(query_embeddings):
            results = await vector_store.search_similar(
                query_embedding=embedding,
                top_k=8  # Increased for better coverage
            )
            # Add query context
            for result in results:
                result['query_index'] = i
                result['original_query'] = query_analysis['expanded_queries'][i]
            all_search_results.extend(results)
        
        # Step 4: Enhanced result merging
        search_results = merge_search_results(all_search_results)
        
        # Step 5: Dynamic relevance filtering
        threshold = calculate_dynamic_threshold(query_analysis, search_results)
        relevant_results = [r for r in search_results if r['score'] >= threshold]
        
        logger.info(f"📋 Found {len(relevant_results)} relevant chunks (threshold: {threshold:.2f})")
        
        if not relevant_results:
            # Create no results response
            suggestions = [
                "Try rephrasing your question",
                "Ask about specific Air Force topics like leadership, training, or heritage",
                "Use more specific terms or acronyms",
                "Check if your question is related to Air Force policies and procedures"
            ]
            
            answer = f"""I couldn't find specific information about "{request.question}" in the Air Force documentation.

**Suggestions:**
{chr(10).join(f'• {suggestion}' for suggestion in suggestions)}

**Topics I can help with:**
• Air Force mission and heritage
• Leadership responsibilities  
• Training and education
• Military procedures and policies
• Aviation history and flying aces"""

            return AnswerResponse(
                answer=answer,
                confidence=ConfidenceLevel.LOW,
                relevance_score=0.0,
                sources=[],
                processing_time=round(time.time() - start_time, 2),
                chunks_found=0
            )
        
        # Step 6: Prepare enhanced context
        top_results = relevant_results[:request.max_sources]
        context_chunks = [
            {
                'content': result['text'],
                'metadata': {
                    'source': result['source'],
                    'doc_type': result['doc_type'],
                    'chunk_id': result['chunk_id'],
                    'relevance_score': result['score']
                }
            }
            for result in top_results
        ]
        
        # Step 7: Enhanced LLM response generation
        try:
            if await llm_service.is_available():
                enhanced_prompt = enhance_llm_prompt(request.question, context_chunks, query_analysis)
                answer_text = await llm_service.generate_response(
                    user_question=request.question,
                    context_chunks=context_chunks
                )
            else:
                # Fallback answer
                answer_parts = [f"Based on Air Force documentation, here's what I found regarding '{request.question}':"]
                for i, chunk in enumerate(context_chunks[:3], 1):
                    content = chunk['content']
                    source = chunk['metadata']['source']
                    content = content[:300] + "..." if len(content) > 300 else content
                    answer_parts.append(f"\n**Source {i}** ({source}):\n{content}")
                answer_text = "\n".join(answer_parts)
        except Exception as llm_error:
            logger.error(f"❌ LLM generation failed: {str(llm_error)}")
            # Fallback answer
            answer_parts = [f"Based on Air Force documentation, here's what I found regarding '{request.question}':"]
            for i, chunk in enumerate(context_chunks[:3], 1):
                content = chunk['content']
                source = chunk['metadata']['source']
                content = content[:300] + "..." if len(content) > 300 else content
                answer_parts.append(f"\n**Source {i}** ({source}):\n{content}")
            answer_text = "\n".join(answer_parts)
        
        # Step 8: Quality assessment
        relevance_scores = [r['score'] for r in top_results]
        quality = assess_answer_quality(answer_text, relevance_scores)
        
        # Step 9: Create enhanced source citations
        sources = []
        if request.include_sources:
            for result in top_results:
                confidence = determine_confidence_level(result['score'])
                sources.append(SourceCitation(
                    document=result['source'],
                    doc_type=result['doc_type'],
                    chunk_id=result['chunk_id'],
                    chapter_number=result.get('chapter_number'),
                    chapter_title=result.get('chapter_title'),
                    section_id=result.get('section_id'),
                    section_title=result.get('section_title'),
                    start_page=result.get('start_page'),
                    end_page=result.get('end_page'),
                    edition=result.get('edition'),
                    granularity=result.get('granularity', 'section_chunk'),
                    relevance_score=round(result['score'], 3),
                    confidence=confidence,
                    preview=result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                ))
        
        processing_time = time.time() - start_time
        overall_confidence = determine_confidence_level(quality['confidence_score'])
        
        # Add confidence indicator to answer
        if overall_confidence == ConfidenceLevel.HIGH:
            formatted_answer = f"🟢 **High Confidence Answer**\n\n{answer_text}"
        elif overall_confidence == ConfidenceLevel.MEDIUM:
            formatted_answer = f"🟡 **Medium Confidence Answer**\n\n{answer_text}"
        else:
            formatted_answer = f"🔴 **Low Confidence Answer** - Limited information available\n\n{answer_text}"
        
        logger.info(f"✅ Question answered in {processing_time:.2f}s with {overall_confidence} confidence")
        
        return AnswerResponse(
            answer=formatted_answer,
            confidence=overall_confidence,
            relevance_score=round(max(relevance_scores), 3),
            sources=sources,
            processing_time=round(processing_time, 2),
            chunks_found=len(relevant_results)
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the Air Force RAG chatbot.
    
    This endpoint:
    1. Takes a user question
    2. Converts it to an embedding
    3. Searches for relevant Air Force documents
    4. Builds a response based on found documents
    
    Args:
        request: Chat request with user message and optional history
        
    Returns:
        AI response with source citations
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"💬 Processing chat request: {request.message[:100]}...")
        
        # Step 1: Expand query for better semantic matching
        expanded_queries = QueryProcessor.expand_air_force_query(request.message)
        logger.info(f"🔍 Expanded to {len(expanded_queries)} queries: {expanded_queries[:3]}")
        
        # Step 2: Get embeddings for all expanded queries
        query_embeddings = await asyncio.gather(*[
            embedding_service.get_embedding(query) for query in expanded_queries
        ])
        
        # Step 3: Search with multiple query embeddings and merge results
        all_search_results = []
        for i, embedding in enumerate(query_embeddings):
            results = await vector_store.search_similar(
                query_embedding=embedding,
                top_k=5  # Reduced per query
            )
            # Add query context to results
            for result in results:
                result['query_index'] = i
                result['original_query'] = expanded_queries[i]
            all_search_results.extend(results)
        
        # Step 4: Merge and deduplicate results
        search_results = merge_search_results(all_search_results)
        
        # Step 5: Dynamic relevance threshold based on query type
        threshold = 0.4  # Default threshold
        
        # Lower threshold for historical/aviation queries which may be more specific
        query_lower = request.message.lower()
        if any(term in query_lower for term in ['flying aces', 'aces', 'aviation history', 'pilots', 'combat']):
            threshold = 0.3  # More lenient for aviation topics
            logger.info(f"🛩️ Using aviation threshold: {threshold}")
        
        relevant_results = [
            result for result in search_results 
            if result['score'] >= threshold
        ]
        
        # Step 6: Prepare context chunks for LLM
        sources = []
        context_chunks = []
        
        if relevant_results:
            # Take top 5 most relevant results for better context
            top_results = relevant_results[:5]
            
            for result in top_results:
                # Prepare context chunk for LLM
                context_chunks.append({
                    'content': result['text'],
                    'metadata': {
                        'source': result['source'],
                        'doc_type': result['doc_type'],
                        'chunk_id': result['chunk_id'],
                        'page': result.get('page', 'Unknown')
                    }
                })
                
                # Collect source information for response
                sources.append({
                    "document": result['source'],
                    "doc_type": result['doc_type'],
                    "chunk_id": result['chunk_id'],
                    "similarity_score": round(result['score'], 3),
                    "page": result.get('page', 'Unknown')
                })
        
        # Step 7: Generate response using LLM
        try:
            # Check if Ollama is available
            if await llm_service.is_available():
                logger.info("🦙 Using LLM for response generation")
                response_text = await llm_service.generate_response(
                    user_question=request.message,
                    context_chunks=context_chunks
                )
            else:
                logger.warning("⚠️ LLM unavailable, using fallback response")
                # Create formatted fallback response with green checkmarks
                if context_chunks:
                    context_parts = []
                    for i, chunk in enumerate(context_chunks[:3], 1):
                        source = chunk['metadata']['source']
                        doc_type = chunk['metadata']['doc_type']
                        content = chunk['content']
                        
                        # Extract roles and format them on separate lines
                        formatted_content = content
                        # Look for role patterns and add line breaks (without checkmarks)
                        formatted_content = re.sub(r'(\d+\.\d+\.)', r'\n\1', formatted_content)
                        # Clean up extra whitespace
                        formatted_content = re.sub(r'\n\s*\n', '\n', formatted_content)
                        
                        context_parts.append(f"**Source {i}** ({doc_type} - {source}):\n{formatted_content}")
                    
                    response_text = (
                        f"Based on Air Force documentation, here's what I found regarding: **{request.message}**\n\n"
                        + "\n\n---\n\n".join(context_parts)
                    )
                else:
                    response_text = llm_service._create_fallback_response(
                        user_question=request.message,
                        context_chunks=context_chunks
                    )
        except Exception as llm_error:
            logger.error(f"❌ LLM generation failed: {str(llm_error)}")
            # Fallback to simple response
            if context_chunks:
                context_parts = []
                for i, chunk in enumerate(context_chunks[:3], 1):
                    source = chunk['metadata']['source']
                    doc_type = chunk['metadata']['doc_type']
                    content = chunk['content']
                    
                    # Extract roles and format them on separate lines
                    formatted_content = content
                    # Look for role patterns and add line breaks (without checkmarks)
                    formatted_content = re.sub(r'(\d+\.\d+\.)', r'\n\1', formatted_content)
                    # Clean up extra whitespace
                    formatted_content = re.sub(r'\n\s*\n', '\n', formatted_content)
                    
                    context_parts.append(f"**Source {i}** ({doc_type} - {source}):\n{formatted_content}")
                
                response_text = (
                    f"Based on Air Force documentation, here's what I found regarding: **{request.message}**\n\n"
                    + "\n\n---\n\n".join(context_parts) +
                    "\n\n*Note: LLM processing failed - showing raw document excerpts.*"
                )
            else:
                response_text = (
                    f"I couldn't find specific information about '{request.message}' "
                    "in the Air Force roles and responsibilities documentation. "
                    "Try rephrasing your question or asking about specific positions, "
                    "commands, or organizational responsibilities."
                )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Chat response generated in {processing_time:.2f}s with {len(sources)} sources")
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search endpoint for finding relevant Air Force documents.
    
    This endpoint allows direct searching without generating a chat response.
    Useful for exploring available documents or debugging.
    
    Args:
        request: Search request with query and optional filters
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        logger.info(f"🔍 Processing search request: {request.query}")
        
        # Convert search query to embedding
        query_embedding = await embedding_service.get_embedding(request.query)
        
        # Perform search with optional filters
        if request.doc_types:
            search_results = await vector_store.search_with_text_filter(
                query_embedding=query_embedding,
                doc_types=request.doc_types,
                top_k=request.top_k
            )
        else:
            search_results = await vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=request.top_k
            )
        
        logger.info(f"✅ Found {len(search_results)} documents for search query")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"❌ Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/process-pdf", response_model=ProcessPDFResponse)
async def process_pdf(request: ProcessPDFRequest, background_tasks: BackgroundTasks):
    """
    Process local Air Force PDF document and store it in the vector database.
    
    This endpoint:
    1. Loads local PDF file
    2. Extracts roles & responsibilities sections
    3. Creates embeddings for text chunks
    4. Stores everything in Pinecone
    
    Args:
        request: Local PDF file path to process
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Processing summary with statistics
    """
    import time
    start_time = time.time()
    
    try:
        pdf_file = request.pdf_file
        logger.info(f"📄 Starting PDF processing for local file: {pdf_file}")
        
        # Check if file exists
        if not os.path.exists(pdf_file):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_file}")
        
        total_chunks = 0
        successful = 0
        failed = 0
        
        try:
            logger.info(f"📄 Processing local PDF: {pdf_file}")
            
            # Step 1: Extract and chunk PDF content from local file
            documents = pdf_processor.process_pdf(pdf_file)
            
            if not documents:
                logger.warning(f"⚠️ No content extracted from {pdf_file}")
                failed += 1
            else:
                # Step 2: Create embeddings for all chunks
                texts = [doc["text"] for doc in documents]
                embeddings = await embedding_service.get_embeddings_batch(texts)
                
                # Step 3: Store in vector database
                for doc, embedding in zip(documents, embeddings):
                    await vector_store.upsert_document(
                        text=doc["text"],
                        embedding=embedding,
                        metadata={
                            "source": doc["source"],
                            "chunk_id": doc["chunk_id"],
                            "doc_type": doc["doc_type"],
                            "chapter_number": doc["chapter_number"],
                            "chapter_title": doc["chapter_title"],
                            "section_id": doc["section_id"],
                            "section_title": doc["section_title"],
                            "start_page": doc["start_page"],
                            "end_page": doc["end_page"],
                            "edition": doc["edition"],
                            "granularity": doc["granularity"],
                            "total_chunks": doc["total_chunks"]
                        }
                    )
                
                total_chunks += len(documents)
                successful += 1
                
                logger.info(f"✅ Successfully processed {pdf_file} - {len(documents)} chunks")
                
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file}: {str(e)}")
            failed += 1
        
        processing_time = time.time() - start_time
        
        logger.info(f"🎉 PDF processing complete! {successful}/1 successful")
        
        return ProcessPDFResponse(
            total_files=1,
            total_chunks=total_chunks,
            successful=successful,
            failed=failed,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF processing endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/api/process-local-afh1")
async def process_local_afh1():
    """
    Process the local afh1.pdf file.
    
    This is a convenience endpoint that processes the local afh1.pdf file.
    Use this to populate the database with AFH1 content.
    """
    # Process the local AFH1 file
    request = ProcessPDFRequest(pdf_file="afh1.pdf")
    return await process_pdf(request, BackgroundTasks())


@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics and information.
    
    Returns:
        Current system status and database statistics
    """
    try:
        # Get vector database stats
        vector_stats = await vector_store.get_index_stats()
        
        # Get embedding service info
        embedding_info = embedding_service.get_model_info()
        
        # Get PDF processor info
        processor_info = pdf_processor.get_processor_stats()
        
        return {
            "database": {
                "total_documents": vector_stats.get("total_vectors", 0),
                "index_fullness": vector_stats.get("index_fullness", 0),
                "dimension": vector_stats.get("dimension", 0)
            },
            "embedding_service": embedding_info,
            "pdf_processor": processor_info,
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"❌ Stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/api/test-qa")
async def test_qa_system():
    """
    Test endpoint for the updated Q&A system with sample Air Force questions.
    
    Returns:
        Test results with various question types
    """
    try:
        # Sample Air Force questions for testing
        test_questions = [
            "What is the Air Force mission?",
            "How do I submit a leave request?",
            "What are the requirements for promotion?",
            "What is the dress code for uniforms?",
            "How does the disciplinary process work?",
            "What benefits are available to airmen?",
            "What is the purpose of BMT?",
            "How do deployments work?"
        ]
        
        test_results = []
        
        for question in test_questions[:3]:  # Test first 3 to avoid overwhelming
            try:
                # Test query expansion
                expanded_queries = QueryProcessor.expand_air_force_query(question)
                
                # Test search
                query_embedding = await embedding_service.get_embedding(question)
                search_results = await vector_store.search_similar(
                    query_embedding=query_embedding,
                    top_k=3
                )
                
                test_results.append({
                    "question": question,
                    "expanded_queries": expanded_queries[:3],
                    "search_results_count": len(search_results),
                    "top_similarity_score": search_results[0]['score'] if search_results else 0,
                    "status": "success"
                })
                
            except Exception as e:
                test_results.append({
                    "question": question,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "message": "Q&A system test completed",
            "total_questions_tested": len(test_results),
            "results": test_results,
            "system_status": "operational" if all(r.get("status") == "success" for r in test_results) else "issues_detected"
        }
        
    except Exception as e:
        logger.error(f"❌ Test Q&A system error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test Q&A system: {str(e)}")


# ================================
# Startup and Shutdown Events
# ================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize services when the API starts.
    
    This runs once when the server starts up.
    """
    logger.info("🚀 Starting Air Force RAG API...")
    logger.info("✅ All services initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup when the API shuts down.
    
    This runs when the server is stopped.
    """
    logger.info("🛑 Shutting down Air Force RAG API...")
    logger.info("✅ Cleanup completed!")


@app.delete("/api/clear-index")
async def clear_index():
    """
    Clear all vectors from the Pinecone index.
    
    ⚠️ WARNING: This will delete ALL documents from the vector database!
    Use this when you want to start fresh with new PDF processing.
    
    Returns:
        Confirmation of deletion
    """
    import time
    start_time = time.time()
    
    try:
        logger.info("🗑️ Clear index request received")
        
        # Get current stats
        stats = await vector_store.get_index_stats()
        initial_count = stats.get('total_vectors', 0)
        
        logger.info(f"📊 Current vectors in index: {initial_count}")
        
        if initial_count == 0:
            return {
                "message": "Index is already empty",
                "vectors_deleted": 0,
                "processing_time": round(time.time() - start_time, 2)
            }
        
        # Clear the index
        success = await vector_store.clear_all_vectors()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear index")
        
        # Check final stats
        final_stats = await vector_store.get_index_stats()
        final_count = final_stats.get('total_vectors', 0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Index cleared in {processing_time:.2f}s")
        logger.info(f"📊 Vectors deleted: {initial_count - final_count}")
        
        return {
            "message": "Index successfully cleared",
            "initial_vectors": initial_count,
            "final_vectors": final_count,
            "vectors_deleted": initial_count - final_count,
            "processing_time": round(processing_time, 2),
            "note": "It may take a few moments for all vectors to be completely removed from Pinecone"
        }
        
    except Exception as e:
        logger.error(f"❌ Clear index error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    # Run the server
    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",  # Import path to the FastAPI app
        host=host,
        port=port,
        reload=debug,  # Auto-reload on code changes in debug mode
        log_level="info"
    )

"""
API Usage Examples:

1. Health Check:
   GET http://localhost:8000/health

2. Chat with the bot:
   POST http://localhost:8000/api/ask
   Body: {"message": "What is the Air Force mission?"}

3. Search documents:
   POST http://localhost:8000/api/search  
   Body: {"query": "leadership responsibilities", "top_k": 5}

4. Process local PDF:
   POST http://localhost:8000/api/process-pdf
   Body: {"pdf_file": "afh1.pdf"}

5. Process AFH1 (convenience):
   POST http://localhost:8000/api/process-local-afh1

6. Get system stats:
   GET http://localhost:8000/api/stats

7. Clear vector database:
   DELETE http://localhost:8000/api/clear-index

8. API Documentation:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)
"""