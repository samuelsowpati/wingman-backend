# Air Force RAG Chatbot Backend

A FastAPI-based backend for an AI chatbot that answers questions about Air Force roles and responsibilities using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Pinecone account and API key
- Internet connection (for downloading PDFs and AI models)

### Installation

1. **Clone and navigate to the project:**
```bash
cd wingman2
```

2. **Create a virtual environment:**

**On Windows (PowerShell/Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Copy the example file
cp env_example.txt .env

# Edit .env with your actual values
PINECONE_API_KEY=your_pinecone_api_key_here
```

5. **Run the server:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs` (interactive API documentation)
- **ReDoc**: `http://localhost:8000/redoc` (alternative documentation)

## 🏗️ Architecture Overview

```
User Question → FastAPI → Embedding Service → Pinecone Search → Context Assembly → Response
```

### Core Components

1. **FastAPI Application** (`main.py`)
   - Web API server
   - Request/response handling
   - CORS configuration for frontend

2. **Embedding Service** (`embeddings.py`)
   - Converts text to vector embeddings
   - Uses Sentence Transformers locally
   - Creates 384-dimensional vectors

3. **PDF Processor** (`pdf_processor.py`)
   - Downloads Air Force PDFs
   - Extracts "Roles & Responsibilities" sections
   - Chunks text for optimal search

4. **Vector Store** (`vector_store.py`)
   - Manages Pinecone vector database
   - Stores and searches document embeddings
   - Handles metadata and filtering

## 🔧 Key API Endpoints

### Health Check
```bash
GET /health
```
Returns system status and service information.

### Chat
```bash
POST /api/chat
{
  "message": "What does SAF/AQ do?",
  "history": []
}
```
Main chatbot endpoint that answers questions about Air Force roles.

### Search Documents
```bash
POST /api/search
{
  "query": "acquisition responsibilities",
  "top_k": 5,
  "doc_types": ["AFI"]
}
```
Search for relevant documents without generating a chat response.

### Process PDFs
```bash
POST /api/process-pdfs
{
  "pdf_urls": [
    "https://static.e-publishing.af.mil/.../afi10-2402.pdf"
  ]
}
```
Process new Air Force PDFs and add them to the knowledge base.

### Batch Process All PDFs
```bash
POST /api/batch-process
```
Process all predefined Air Force PDFs (takes several hours).

## 🔍 How RAG Works

1. **Document Processing**:
   - PDFs are downloaded and text extracted
   - "Roles & Responsibilities" sections are identified
   - Text is chunked into 500-word segments with 50-word overlap

2. **Embedding Creation**:
   - Each text chunk is converted to a 384-dimensional vector
   - Uses `all-MiniLM-L6-v2` Sentence Transformer model
   - Embeddings capture semantic meaning

3. **Vector Storage**:
   - Embeddings stored in Pinecone with metadata
   - Metadata includes source, document type, page numbers
   - Enables fast similarity search

4. **Query Processing**:
   - User questions converted to embeddings
   - Cosine similarity search finds relevant documents
   - Top results filtered by relevance threshold (>0.6)

5. **Response Generation**:
   - Relevant documents provide context
   - Simple template-based response assembly
   - Source citations included

## 🗂️ File Structure

```
wingman2/
├── main.py              # FastAPI application and endpoints
├── embeddings.py        # Text-to-vector conversion service
├── pdf_processor.py     # PDF download and processing
├── vector_store.py      # Pinecone vector database interface
├── requirements.txt     # Python dependencies
├── env_example.txt      # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with these variables:

```bash
# Required
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Model Configuration

The system uses these AI models by default:
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions, good balance of speed/quality)
- **Vector Search**: Cosine similarity in Pinecone
- **Response**: Template-based assembly (no LLM for MVP)

## 📊 Monitoring and Debugging

### System Stats
```bash
GET /api/stats
```
Returns database statistics and system information.

### Logs
The application logs important events:
- PDF processing progress
- Search operations
- Errors and warnings
- Performance metrics

### Database Management
```python
# Check Pinecone index stats
stats = await vector_store.get_index_stats()
print(f"Total documents: {stats['total_vectors']}")

# Clear database (use with caution!)
await vector_store.clear_index()
```

## 🚨 Troubleshooting

### Common Issues

1. **"PINECONE_API_KEY not set"**
   - Create `.env` file with your Pinecone API key
   - Get key from https://pinecone.io

2. **"Failed to download PDF"**
   - Check internet connection
   - Verify PDF URLs are accessible
   - Some PDFs may be protected or moved

3. **"No roles section found"**
   - PDF may not contain "Roles and Responsibilities" section
   - Section may be formatted differently
   - Check PDF processor logs for details

4. **Slow embedding creation**
   - First run downloads ML models (normal)
   - Subsequent runs should be faster
   - Consider using GPU if available

5. **CORS errors from frontend**
   - Add your frontend URL to `ALLOWED_ORIGINS`
   - Default supports React dev servers on ports 3000 and 5173

### Performance Tips

1. **Batch Processing**: Use `/api/process-pdfs` with multiple URLs
2. **Caching**: Embedding models are cached after first load
3. **Filtering**: Use document type filters for faster searches
4. **Chunk Size**: Adjust in `pdf_processor.py` if needed

## 🔮 Future Enhancements

Current implementation is an MVP. Potential improvements:

1. **Advanced LLM Integration**
   - Add Ollama + Llama 3.1 for better responses
   - Streaming responses for real-time experience

2. **Improved Reranking**
   - Add cross-encoder reranking for better relevance
   - Implement hybrid search (vector + keyword)

3. **Enhanced Processing**
   - Better PDF parsing for complex documents
   - Support for tables and images
   - Multi-language document support

4. **Production Features**
   - User authentication and authorization
   - Rate limiting and caching
   - Database backups and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper comments
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the AI4Defense initiative at George Mason University.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check logs for error details
4. Create an issue in the repository

---

**Happy coding! 🚀**