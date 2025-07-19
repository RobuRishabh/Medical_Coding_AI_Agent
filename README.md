# Medical Coding AI Agent
AI-powered system designed to excel at medical coding practice tests using advanced language models, retrieval-augmented generation (RAG), and automated testing frameworks.

## Project Overview
This project creates an intelligent medical coding assistant that can:

- Answer medical coding questions through an interactive chat interface
- Automatically take practice tests by processing PDF documents
- Achieve high scores on CPC (Certified Professional Coder) practice examinations
- Provide detailed reasoning for every answer with source citations

The system combines multiple AI technologies to create a comprehensive medical coding solution that rivals human expert performance.

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                     │
├─────────────────────────────────────────────────────────────────┤
│  Chat Interface              │        Practice Test Runner      │
├─────────────────────────────────────────────────────────────────┤
│                      SmolaGents ToolCallingAgent                │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Base Tool         │         Web Search Tool          │
│  (ChromaDB + Embeddings)     │        (DuckDuckGo API)          │
├─────────────────────────────────────────────────────────────────┤
│                    LiteLLM (GPT-3.5/GPT-4)                      │
├─────────────────────────────────────────────────────────────────┤
│  PDF Processing Pipeline     │      Results & Analytics         │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Infrastructure

### Core AI Components

**AI Agent Framework**: SmolaGents ToolCallingAgent
- **Agent Type**: `ToolCallingAgent` with optimized configuration
- **Max Steps**: 2 (for efficient processing)
- **Planning Interval**: None (streamlined execution)
- **Prompt Templates**: Custom medical coding prompts

**Underlying Language Model**: LiteLLM Integration
- **Primary Model**: GPT-3.5-Turbo / GPT-4 (configurable via `AGENT_MODEL` env var)
- **API Integration**: OpenAI API via LiteLLM wrapper
- **Temperature**: 0.2 (for consistent, focused responses)
- **Timeout**: 3600 seconds for comprehensive reasoning

**Dual-Tool System**:
1. **Knowledge Base Retriever Tool**
   - **Vector Database**: ChromaDB with persistent storage
   - **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
   - **Storage Location**: `Outputs/vector_db/`
   - **Similarity Search**: Top-k retrieval with relevance scoring

2. **Web Search Tool**
   - **Search Engine**: DuckDuckGo Search API
   - **Real-time Updates**: Current medical coding information
   - **Source Verification**: Authoritative medical coding resources

### Knowledge Base Infrastructure

**Document Processing Pipeline**:
```python
# PDF → Markdown Conversion
ConvertPDF2md.py → Outputs/markdown/

# Chunking & Embedding
EmbedChunks2Chroma.py → Outputs/vector_db/
```

**Included Medical Coding Resources**:
- `ICD-10-CM-October-2025-Guidelines.pdf` - Latest ICD-10 guidelines
- `HCPCS.pdf` - Healthcare Common Procedure Coding System
- `cpc_study_guide_sample_pages.pdf` - CPC certification materials
- `AAPC_workshop_2023.pdf` - Professional workshop content

**Vector Database Configuration**:
- **Database**: ChromaDB with SQLite backend
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Chunk Size**: 500 characters with 50 character overlap
- **Storage**: Persistent storage in `Outputs/vector_db/chroma.sqlite3`

### Test Processing Infrastructure

**AutomatedTestRunner Class**:
- **Processing Modes**: 
  - Individual question processing with retry logic
  - Batch processing for efficiency
  - Parallel batch processing for speed
- **Rate Limiting**: 1-second delays between questions
- **Retry Mechanism**: 3 attempts with exponential backoff
- **Tool Strategies**: Fallback tool combinations for reliability

**PDF Processing Pipeline**:
```python
TestProcessor → Question Extraction → Answer Parsing → Validation
```

## Project Structure
```
Medical_Coding_AI_Agent/
├── app.py                          # Main Streamlit application
├── prompts.json                    # AI prompt configurations
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys)
├── data/                          # Source PDF documents
│   ├── AAPC_workshop_2023.pdf
│   ├── cpc_study_guide_sample_pages.pdf
│   ├── HCPCS.pdf
│   └── ICD-10-CM-October-2025-Guidelines.pdf
├── Outputs/                       # Processed data and results
│   ├── Output.log                 # Application logs
│   ├── markdown/                  # Converted PDF content
│   │   ├── AAPC_workshop_2023.md
│   │   ├── cpc_study_guide_sample_pages.md
│   │   ├── HCPCS.md
│   │   └── ICD-10-CM-October-2025-Guidelines.md
│   └── vector_db/                 # ChromaDB vector database
│       └── chroma.sqlite3
├── scripts/                       # Core processing modules
│   ├── smolagent_tools.py         # AI agent tool implementations
│   ├── test_runner.py             # Automated test execution engine
│   ├── test_processor.py          # PDF parsing and question extraction
│   ├── results_generator.py       # Comprehensive report generation
│   ├── ConvertPDF2md.py          # PDF to Markdown conversion
│   └── EmbedChunks2Chroma.py     # Vector database creation
└── temp_test_data/                # Temporary test processing files
```

## Step-by-Step Setup Guide

### Prerequisites
```bash
Python 3.8+
OpenAI API Key
Git
```

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/RobuRishabh/Medical_Coding_AI_Agent.git
cd Medical_Coding_AI_Agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Create environment file
cp .env.example .env

# Edit .env file with your credentials:
OPENAI_API_KEY=your_openai_api_key_here
AGENT_MODEL=gpt-4.1-mini 
```

### 3. Knowledge Base Setup

**Step 3a: Convert PDFs to Markdown**
```bash
python scripts/ConvertPDF2md.py
```
This converts all PDFs in `data/` folder to markdown format in `Outputs/markdown/`.

**Step 3b: Create Vector Database**
```bash
python scripts/EmbedChunks2Chroma.py
```
This processes markdown files into chunks and creates the ChromaDB vector database.

### 4. Launch Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

## Usage Guide

### Chat Interface

**How to Use:**
1. **Navigate** to "Chat Interface" in the sidebar
2. **Configure Tools**:
   - Web Search: Always enabled for current information
   - Knowledge Base: Toggle to search embedded documents
   - Show Sources: Toggle to display source citations
3. **Ask Questions**: Enter medical coding questions in the text area
4. **Review Answers**: Get detailed responses with reasoning and sources

**Example Questions:**
- "How do I code a chest X-ray with interpretation?"
- "What ICD-10 code should I use for Type 2 diabetes?"
- "What's the difference between CPT 99213 and 99214?"

### Practice Test Runner

**Complete Workflow:**

**Step 1: File Upload and Extraction**
1. Navigate to "Practice Test" in sidebar
2. Upload practice test PDF (questions without answers)
3. Upload answer key PDF (with correct answers)
4. Click "🔍 Extract Questions and Answers"
5. Review extraction validation summary

**Step 2: Agent Configuration and Test Execution**
1. **Configure Agent Settings**:
   - Use Knowledge Base: ✅ (recommended)
   - Use Web Search: ✅ (recommended)
   - Temperature: 0.2 (for consistent responses)
2. Click "🚀 Run Automated Test"
3. Monitor real-time progress
4. Review comprehensive results

**Step 3: Results Analysis**
- View test score and performance metrics
- Download detailed report (Markdown format)
- Download raw results (JSON format)
- Start new test if needed

### Using Cached Data (Development Mode)

**For Faster Testing:**
1. Check "Skip PDF extraction (use cached data)"
2. Click "🔍 Extract Questions and Answers" 
3. Proceed directly to Step 2 configuration
4. This uses pre-processed test data from `Outputs/extraction/`

## Agent Configuration Details

### Optimal Agent Setup
```python
agent_config = {
    'model': 'gpt-4.1-mini'
    'temperature': 0.2,
    'timeout': 3600,
    'max_steps': 2,
    'tools': ['knowledge_base_retriever', 'web_search_tool']
}
```

### Tool Strategy System
The system implements intelligent tool fallback strategies:
1. **Primary**: Both Knowledge Base + Web Search
2. **Fallback 1**: Web Search only
3. **Fallback 2**: Knowledge Base only  
4. **Fallback 3**: Model knowledge only

### Prompt Engineering
**Medical Coding Expert Prompts**:
- Role-based instructions for CPC expertise
- Step-by-step reasoning methodology
- Structured output requirements
- Source citation mandates

## Performance Features

### Rate Limiting & Reliability
- **API Rate Limiting**: 1-second delays between questions
- **Retry Logic**: 3 attempts with exponential backoff
- **Error Handling**: Graceful fallbacks and error recovery
- **Progress Tracking**: Real-time progress updates

### Processing Modes
1. **Individual Processing**: Detailed reasoning for each question
2. **Batch Processing**: Efficient processing of multiple questions
3. **Parallel Batch**: Maximum speed processing

### Results & Analytics

**Comprehensive Reporting Includes**:
- Overall test score and percentage
- Question-by-question analysis
- Detailed reasoning for each answer
- Source citations and references
- Answer distribution validation
- Performance timing metrics
- Agent configuration details

**Export Formats**:
- Detailed Markdown reports
- Raw JSON data
- Extraction validation summaries

## Troubleshooting

### Common Issues

**1. Missing API Key**
```bash
Error: OPENAI_API_KEY not found
Solution: Check .env file configuration
```

**2. Vector Database Not Found**
```bash
Error: ChromaDB not initialized
Solution: Run python scripts/EmbedChunks2Chroma.py
```

**3. PDF Extraction Issues**
```bash
Error: No questions extracted
Solution: Verify PDF format and use "Skip extraction" option
```

### Debug Mode
- Check `Outputs/Output.log` for detailed logs
- Use cached data option for development
- Monitor Streamlit console for real-time errors

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with proper documentation
4. Submit pull request with detailed description

## Support
For questions or issues:
1. Check the troubleshooting section
2. Review application logs in `Outputs/Output.log`
3. Open GitHub issue with detailed description
