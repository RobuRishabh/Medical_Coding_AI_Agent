# Medical Coding AI Agent
A sophisticated AI-powered system designed to excel at medical coding practice tests using advanced language models, retrieval-augmented generation (RAG), and automated testing frameworks.

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
│                      SmolaGents CodeAgent                       │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Base Tool         │         Web Search Tool          │
│  (ChromaDB + Embeddings)     │        (DuckDuckGo API)          │
├─────────────────────────────────────────────────────────────────┤
│                   Google Gemini 2.0 Flash                       │
├─────────────────────────────────────────────────────────────────┤
│  PDF Processing Pipeline     │      Results & Analytics         │
└─────────────────────────────────────────────────────────────────┘
```
## Key Features

1. Dual Interface System
- Chat Interface: Interactive Q&A for medical coding questions
- Practice Test Runner: Automated test-taking with comprehensive scoring

2. Advanced AI Agent
- Framework: SmolaGents CodeAgent for structured reasoning
- Model: Google Gemini 2.0 Flash for state-of-the-art performance
- Tools: Dual-tool system combining knowledge base and web search

3. Intelligent Knowledge Base
- Vector Database: ChromaDB for semantic search
- Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- Documents: ICD-10, CPT, HCPCS, and CPC study materials

4. Automated Test Processing
- PDF Parsing: Converts practice tests to structured data
- Answer Extraction: Intelligent parsing of both questions and answer keys
- Scoring System: Comprehensive evaluation with detailed analytics

## Project Structure
```
Medical_Coding_AI_Agent/
├── app.py                          # Main Streamlit application
├── prompts.json                    # AI prompt configurations
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables
├── data/                          # Source PDF documents
│   ├── AAPC_workshop_2023.pdf
│   ├── cpc_study_guide_sample_pages.pdf
│   ├── HCPCS.pdf
│   └── ICD-10-CM-October-2025-Guidelines.pdf
├── Outputs/                       # Processed data
│   ├── markdown/                  # Converted PDF content
│   └── vector_db/                 # ChromaDB vector database
├── scripts/                       # Core processing modules
│   ├── smolagent_tools.py         # AI agent tools
│   ├── test_runner.py             # Automated test execution
│   ├── test_processor.py          # PDF parsing and processing
│   ├── results_generator.py       # Report generation
│   ├── ConvertPDF2md.py          # PDF to Markdown conversion
│   └── EmbedChunks2Chroma.py     # Vector database creation
└── temp_test_data/                # Temporary test processing files
```

## Infrastructure & Technology Stack
### Core AI Components

- Language Model: Google Gemini 2.0 Flash
    - Temperature: 0.2 for consistent, focused responses
    - Context window: 1M tokens for comprehensive reasoning
    - Multimodal capabilities for diverse input processing
- Agent Framework: SmolaGents CodeAgent
    - Structured reasoning with tool integration
    - Code execution capabilities for complex calculations
    - Multi-step problem solving with retry logic

### Knowledge Management

- Vector Database: ChromaDB
    - Persistent storage for semantic search
    - Optimized for medical coding terminology
    - Real-time similarity matching
- Embeddings: HuggingFace sentence-transformers
    - Model: all-MiniLM-L6-v2
    - 384-dimensional vectors
    - Optimized for semantic similarity
- Document Processing Pipeline
    - PDF Conversion: Custom markdown conversion
    - Text Chunking: Recursive character splitting (500 chars, 50 overlap)
    - Question Parsing: Regex-based pattern matching
    - Answer Extraction: Multi-pattern recognition system
- Web Integration
    - Search Engine: DuckDuckGo API
    - Real-time Updates: Current medical coding information
    - Source Verification: Authoritative medical coding resources

### Knowledge Base Contents
The system includes comprehensive medical coding resources:
1. ICD-10-CM Guidelines (October 2025)
2. HCPCS Level II Codes
3. CPT Procedure Codes
4. CPC Study Materials
5. AAPC Workshop Content
All documents are processed into a searchable vector database for instant retrieval during question answering.

## AI Agent Configuration
### Optimal Agent Setup
```
agent_config = {
    'model': 'gemini-2.0-flash',
    'tools': ['knowledge_base_retriever', 'web_search'],
    'temperature': 0.2,
    'max_steps': 5,
    'timeout': 60
}
```
### Tool Integration
- Knowledge Base Tool: Semantic search across medical coding documents
- Web Search Tool: Real-time information retrieval for current guidelines
- Dual-tool Strategy: Combines local expertise with current information

### Prompt Engineering
Specialized prompts following META_PROMPT structure:
- Role definition as Certified Professional Coder
- Step-by-step reasoning methodology
- Citation requirements for source attribution
- Structured output format for consistency

### 🔄 Practice Test Workflow
1. Document Upload
- Upload practice test PDF (questions only)
- Upload answer key PDF (with correct answers)
2. Processing Pipeline
```
# PDF → Markdown conversion
markdown_text = convert_pdf_to_markdown(pdf_path)

# Question extraction
questions = parse_questions_from_markdown(markdown_text)

# Answer key processing
answers = parse_answers_from_markdown(answer_key_text)
```
3. Agent Execution
- Dynamic agent creation with selected tools
- Rate limiting (4-7 seconds between questions)
- Retry logic for API failures
- Comprehensive reasoning extraction
4. Results Generation
- Detailed question-by-question analysis
- Performance metrics and scoring
- Source attribution for all answers
- Downloadable comprehensive reports

🚀 Getting Started

Prerequisites
```
Python 3.8+
Google Gemini API Key
```
Installation
```
# Clone repository
git clone https://github.com/RobuRishabh/Medical_Coding_AI_Agent.git
cd Medical_Coding_AI_Agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

Initial Setup
```
# Convert PDFs to markdown
python scripts/ConvertPDF2md.py

# Create vector database
python scripts/EmbedChunks2Chroma.py

# Run the application
streamlit run app.py
```
## 📊 Usage Examples
### Chat Interface
1. Navigate to "Chat Interface"
2. Ask medical coding questions
3. Toggle knowledge base and web search
4. View detailed answers with sources
### Practice Test
1. Navigate to "Practice Test"
2. Upload test PDF and answer key
3. Configure agent settings
4. Run automated test
5. Review comprehensive results

## 🔧 Configuration Options
### Agent Settings
- Knowledge Base: Toggle local document search
- Web Search: Enable real-time information retrieval
- Temperature: Control response creativity (0.0-1.0)
### Performance Tuning
- Max Steps: Limit agent reasoning iterations
- Timeout: Set maximum processing time
- Rate Limits: Adjust API call frequency

## 📈 Results & Analytics
- Comprehensive Reporting
- Overall test score and percentage
- Question-by-question analysis
- Detailed reasoning for each answer
- Source citations and references
- Performance metrics and timing

## License
This project is licensed under the MIT License - see the LICENSE file for details.
