import streamlit as st
import os
import json
import logging
import re
from dotenv import load_dotenv
from smolagents import LiteLLMModel
from smolagents import ToolCallingAgent
from scripts.smolagent_tools import knowledge_base_retriever, web_search_tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datetime import datetime
import litellm
from pathlib import Path

# ─────────────────────────────
# LOGGING SETUP
# ─────────────────────────────
outputs_dir = Path("Outputs")
outputs_dir.mkdir(exist_ok=True)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_file_path = outputs_dir / "Output.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file_path), mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("=== Medical Coding AI Agent Application Started ===")
logger.info(f"Log file location: {log_file_path.absolute()}")
logger.info(f"Current working directory: {os.getcwd()}")

# ─────────────────────────────
# STREAMLIT CONFIG & ENV
# ─────────────────────────────
st.set_page_config(
    page_title="CPC Medical Coding Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger.info("Streamlit page config and environment variables set")

# ─────────────────────────────
# ENV VARIABLES
# ─────────────────────────────
load_dotenv()
logger.info("Environment variables loaded")
required_vars = ["OPENAI_API_KEY", "AGENT_MODEL"]
for var in required_vars:
    if os.getenv(var):
        logger.info(f"Environment variable {var} is set")
    else:
        logger.warning(f"Environment variable {var} is not set")

# ─────────────────────────────
# PROMPT LOADER
# ─────────────────────────────
def load_system_prompt(path="prompts.json") -> str:
    logger.info(f"Loading system prompt from {path}")
    try:
        with open(path, "r", encoding="utf-8") as file:
            prompts = json.load(file)
        prompt = prompts.get("CPC_AGENT_PROMPT")
        logger.info("System prompt loaded successfully")
        return prompt
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        st.error(f"Failed to load system prompt: {e}")
    return "You are a medical coding assistant."

# ─────────────────────────────
# AGENT CREATION (NO WRAPPER)
# ─────────────────────────────
def create_fast_agent(use_knowledge_base=True, use_web_search=True):
    logger.info("Creating fast agent with ToolCallingAgent (no wrapper)...")
    try:
        base_prompt = load_system_prompt()
        tool_names = []
        if use_web_search:
            tool_names.append(web_search_tool.name)
        if use_knowledge_base:
            tool_names.append(knowledge_base_retriever.name)
        
        try:
            enhanced_system_prompt = base_prompt.format(tool_names=tool_names)
        except Exception as e:
            logger.warning(f"Error occurred while formatting prompt: {e}")
            enhanced_system_prompt = base_prompt  # fallback

        selected_tools = []
        if use_web_search:
            selected_tools.append(web_search_tool)
        if use_knowledge_base:
            selected_tools.append(knowledge_base_retriever)

        prompt_templates = {
            "system_prompt": enhanced_system_prompt,
            "planning": {
                "initial_plan": "",
                "update_plan_pre_messages": "",
                "update_plan_post_messages": "",
            },
            "managed_agent": {
                "task": "",
                "report": "",
            },
            "final_answer": {
                "pre_messages": "",
                "post_messages": "",
            },
        }
        
        # Create LLM with error handling
        try:
            llm = LiteLLMModel(
                model_id=os.getenv("AGENT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to create LiteLLM model: {e}")
            raise

        agent = ToolCallingAgent(
            tools=selected_tools,
            model=llm,
            max_steps=2,
            planning_interval=None,
            prompt_templates=prompt_templates
        )
        logger.info("Fast ToolCallingAgent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create fast ToolCallingAgent: {e}")
        st.error(f"Failed to create fast ToolCallingAgent: {e}")
        return None

def create_test_optimized_agent():
    logger.info("Creating test-optimized ToolCallingAgent (no wrapper)...")
    try:
        with open("prompts.json", encoding="utf-8") as f:
            test_prompt = json.load(f).get("PRACTICE_TEST_PROMPT")
        selected_tools = [knowledge_base_retriever, web_search_tool]
        prompt_templates = {
            "system_prompt": test_prompt,
            "planning": {
                "initial_plan": "",
                "update_plan_pre_messages": "",
                "update_plan_post_messages": "",
            },
            "managed_agent": {
                "task": "",
                "report": "",
            },
            "final_answer": {
                "pre_messages": "",
                "post_messages": "",
            },
        }
        llm = LiteLLMModel(
            model_id=os.getenv("AGENT_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        agent = ToolCallingAgent(
            tools=selected_tools,
            model=llm,
            max_steps=2,
            planning_interval=None,
            prompt_templates=prompt_templates
        )
        logger.info("Test ToolCallingAgent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create test-optimized ToolCallingAgent: {e}")
        return None

def initialize_base_agent():
    logger.info("Initializing base agent...")
    try:
        system_prompt = load_system_prompt()
        logger.info(f"knowledge_base_retriever: {knowledge_base_retriever}")
        logger.info(f"web_search_tool: {web_search_tool}")
        if hasattr(knowledge_base_retriever, 'name'):
            logger.info(f"Knowledge base tool name: {knowledge_base_retriever.name}")
        if hasattr(web_search_tool, 'name'):
            logger.info(f"Web search tool name: {web_search_tool.name}")
        logger.info("Base agent components created successfully")
        return os.getenv("AGENT_MODEL", "gpt-3.5-turbo"), system_prompt
    except Exception as e:
        logger.error(f"Failed to create base agent: {e}")
        st.error(f"Failed to create base agent: {e}")
        return None, None

# ─────────────────────────────
# UTILITIES (CITATIONS, ETC.)
# ─────────────────────────────
def extract_citations_from_response(response_text):
    """Extract citations from response text using regex patterns"""
    citations = {'web_sources': [], 'knowledge_base_sources': []}
    
    if "**Sources:**" in response_text:
        sources_section = response_text.split("**Sources:**")[1] if "**Sources:**" in response_text else ""
        
        # Extract web search citations
        web_pattern = r'- Web Search: \[([^\]]+)\]\(([^)]+)\)(.*)'
        web_matches = re.findall(web_pattern, sources_section)
        for match in web_matches:
            citations['web_sources'].append({
                'name': match[0],
                'url': match[1],
                'description': match[2].strip(' -')
            })
        
        # Extract knowledge base citations
        kb_pattern = r'- Knowledge Base: ([^,]+), Section: "([^"]+)"(.*)'
        kb_matches = re.findall(kb_pattern, sources_section)
        for match in kb_matches:
            citations['knowledge_base_sources'].append({
                'document': match[0],
                'section': match[1],
                'description': match[2].strip(' -')
            })
    
    return citations

def display_enhanced_sources(response_text, use_knowledge_base, use_web_search):
    st.markdown("---")
    st.subheader("🔧 Tools Used & Sources")
    
    # Show tools configuration
    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        if use_knowledge_base:
            st.success("✅ Knowledge Base Used")
        else:
            st.info("➖ Knowledge Base Not Used")
    with col_ref2:
        if use_web_search:
            st.success("✅ Web Search Used")
        else:
            st.info("➖ Web Search Not Used")
    
    # Extract and display actual sources from response
    citations = extract_citations_from_response(response_text)
    
    if citations['web_sources'] or citations['knowledge_base_sources']:
        st.markdown("### 📚 Sources Found")
        
        # Display web sources
        if citations['web_sources']:
            st.markdown("#### 🌐 Web Sources:")
            for source in citations['web_sources']:
                with st.expander(f"🔗 {source['name']}"):
                    st.write(f"**URL:** {source['url']}")
                    if source['description']:
                        st.write(f"**Description:** {source['description']}")
        
        # Display knowledge base sources
        if citations['knowledge_base_sources']:
            st.markdown("#### 📖 Knowledge Base Sources:")
            for source in citations['knowledge_base_sources']:
                with st.expander(f"📄 {source['document']}"):
                    st.write(f"**Section:** {source['section']}")
                    if source['description']:
                        st.write(f"**Description:** {source['description']}")
    else:
        # Check if response contains any source indicators
        if "**Sources:**" in response_text or "Sources:" in response_text:
            st.info("📋 Sources mentioned in response but couldn't parse them automatically")
            with st.expander("View raw sources section"):
                # Try to extract the sources section manually
                if "**Sources:**" in response_text:
                    sources_section = response_text.split("**Sources:**")[1]
                elif "Sources:" in response_text:
                    sources_section = response_text.split("Sources:")[1]
                else:
                    sources_section = "Could not extract sources section"
                st.code(sources_section)
        else:
            st.info("📋 No explicit sources found in response")

def safe_extract_response(raw_response):
    """Safely extract response content with better error handling"""
    try:
        # Handle string responses
        if isinstance(raw_response, str):
            return raw_response
        
        # Handle dict responses (API format)
        if isinstance(raw_response, dict):
            if 'content' in raw_response:
                return raw_response['content']
            elif 'choices' in raw_response and len(raw_response['choices']) > 0:
                choice = raw_response['choices'][0]
                if isinstance(choice, dict) and 'message' in choice:
                    return choice['message'].get('content', str(raw_response))
                return str(choice)
            return str(raw_response)
        
        # Handle objects with content attribute
        if hasattr(raw_response, 'content'):
            content = raw_response.content
            return str(content) if content is not None else str(raw_response)
        
        # Handle message-like objects (LiteLLM responses)
        if hasattr(raw_response, 'choices') and raw_response.choices:
            try:
                first_choice = raw_response.choices[0]
                if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                    content = first_choice.message.content
                    return str(content) if content is not None else str(raw_response)
            except (IndexError, AttributeError):
                pass
        
        # Handle other response formats
        if hasattr(raw_response, 'text'):
            return str(raw_response.text)
        
        # Last resort - convert to string but log warning
        logger.warning(f"Unknown response type: {type(raw_response)}, converting to string")
        return str(raw_response)
        
    except Exception as e:
        logger.error(f"Error extracting response: {e}, response type: {type(raw_response)}")
        return str(raw_response)

def clean_response_text(response_text):
    return response_text.strip()

# ─────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────
def chat_interface():
    logger.info("Rendering chat interface")
    st.title("🩺 CPC Medical Coding Assistant - Chat Interface")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("💬 Ask Your Question")
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        question = st.text_area(
            "Enter your medical coding question:",
            placeholder="e.g., What is the CPT code for a routine colonoscopy?",
            height=100,
            value=st.session_state.current_question,
            key="chat_question_input"
        )
        if question != st.session_state.current_question:
            st.session_state.current_question = question
        st.subheader("💡 Example Questions")
        examples = [
            "How do I code a chest X-ray with interpretation?",
            "What ICD-10 code should I use for Type 2 diabetes?",
            "What's the difference between CPT 99213 and 99214?",
        ]
        for i, example in enumerate(examples):
            if st.button(f"📝 {example}", key=f"chat_example_{i}"):
                st.session_state.current_question = example
                st.rerun()
    with col2:
        st.header("⚙️ Options")
        st.markdown("🌐 **Web Search**: Always Enabled")
        st.info("Web search is always active to provide the most current information.")
        use_web_search = True
        use_knowledge_base = st.checkbox("🔍 Search Knowledge Base", value=True, key="chat_use_kb")
        st.info("Knowledge base search is optional. Uncheck to skip searching the embedded documents.")
        show_sources = st.checkbox("📖 Show Sources", value=True, key="chat_show_sources")
    if st.button("🚀 Get Answer", type="primary", use_container_width=True, key="chat_get_answer"):
        if not st.session_state.current_question.strip():
            st.warning("Please enter a question!")
            logger.warning("User tried to submit empty question")
            return
        logger.info(f"User submitted question: {st.session_state.current_question[:100]}...")
        llm, system_prompt = initialize_base_agent()
        if not llm or not system_prompt:
            st.error("❌ Agent Failed to Initialize")
            logger.error("Failed to initialize base agent for chat")
            return
        with st.spinner("🔄 Processing your question..."):
            try:
                logger.info(f"Processing query: {st.session_state.current_question}")
                logger.info(f"Using tools - Knowledge Base: {use_knowledge_base}, Web Search: {use_web_search}")
                agent = create_fast_agent(use_knowledge_base, use_web_search)
                if not agent:
                    st.error("Failed to create agent with selected tools")
                    logger.error("Failed to create fast agent")
                    return
                raw_response = agent.run(st.session_state.current_question)
                response_text = safe_extract_response(raw_response)
                response_text = clean_response_text(response_text)
                st.markdown("---")
                st.header("🤖 Answer")
                st.markdown(response_text)
                if show_sources:
                    display_enhanced_sources(response_text, use_knowledge_base, use_web_search)
                logger.info("Query processed successfully")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"⚠️ Error: {str(e)}")
                st.info("Please try rephrasing your question or check the system logs.")


def practice_test_interface():
    """Interface for automated practice test"""
    logger.info("Rendering practice test interface")
    st.title("🩺 CPC Medical Coding Assistant - Practice Test")
    st.markdown("---")
    
    st.header("🎯 Automated Practice Test Runner")
    
    # Initialize session state for test workflow
    if 'test_workflow_state' not in st.session_state:
        st.session_state.test_workflow_state = {
            'step': 1,
            'questions_extracted': False,
            'answers_extracted': False,
            'extraction_results': None,
            'test_completed': False
        }
        logger.info("Initialized test workflow state")
    
    # Step 1: File Upload and Extraction
    st.subheader("📁 Step 1: Upload Files and Extract Questions/Answers")
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        test_file = st.file_uploader(
            "Upload Practice Test (PDF)", 
            type=['pdf'],
            help="Upload the practice test without answers",
            key="test_file_upload"
        )
    
    with col2:
        answers_file = st.file_uploader(
            "Upload Answer Key (PDF)", 
            type=['pdf'],
            help="Upload the practice test with answers",
            key="answers_file_upload"
        )
    
    # Add checkbox to skip extraction for development
    skip_extraction = st.checkbox("Skip PDF extraction (use cached data)", value=False, key="skip_extraction_cb")
    
    # Step 1 Button: Extract Questions and Answers
    extract_button_disabled = not skip_extraction and (not test_file or not answers_file)
    
    if st.button("🔍 Extract Questions and Answers", 
                 type="primary", 
                 disabled=extract_button_disabled,
                 key="extract_questions_answers"):
        
        if skip_extraction:
            logger.info("User chose to skip extraction, using cached data")
            st.info("Using cached test data...")
            st.session_state.test_workflow_state['questions_extracted'] = True
            st.session_state.test_workflow_state['answers_extracted'] = True
            st.session_state.test_workflow_state['step'] = 2
            st.success("✅ Using cached data - Ready for automated test!")
        else:
            logger.info("Starting PDF extraction process")
            # Save uploaded files
            test_path = f"temp_test_{datetime.now().timestamp()}.pdf"
            answers_path = f"temp_answers_{datetime.now().timestamp()}.pdf"
            
            with open(test_path, "wb") as f:
                f.write(test_file.read())
            with open(answers_path, "wb") as f:
                f.write(answers_file.read())
            
            logger.info(f"Saved uploaded files: {test_path}, {answers_path}")
            
            # Extract questions and answers
            with st.spinner("🔄 Extracting questions and answers from PDFs..."):
                try:
                    # Import and initialize test processor
                    from scripts.test_processor import TestProcessor
                    
                    processor = TestProcessor()

                    # Extract questions and answers using the processor methods
                    questions_data = processor.extract_questions_from_pdf(test_path)
                    answers_data = processor.extract_answers_from_pdf(answers_path)

                    # Store extraction results
                    st.session_state.test_workflow_state['extraction_results'] = {
                        'questions': questions_data,
                        'answers': answers_data,
                        'test_path': test_path,
                        'answers_path': answers_path
                    }
                    
                    st.session_state.test_workflow_state['questions_extracted'] = True
                    st.session_state.test_workflow_state['answers_extracted'] = True
                    st.session_state.test_workflow_state['step'] = 2
                    
                    # Display extraction summary
                    st.success("✅ Extraction completed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Questions Extracted", len(questions_data) if questions_data else 0)
                    with col2:
                        st.metric("Answers Extracted", len(answers_data) if answers_data else 0)
                    
                    # Show a preview of extracted data
                    if questions_data and len(questions_data) > 0:
                        with st.expander("📋 Preview Extracted Questions (First 3)"):
                            for i, question in enumerate(questions_data[:3]):
                                st.write(f"**Q{i+1}:** {question['question'][:200]}...")
                                if question.get('options'):
                                    for option in question['options']:
                                        st.write(f"  - {option}")
                    
                    if answers_data and len(answers_data) > 0:
                        with st.expander("📝 Preview Extracted Answers (First 3)"):
                            for i, answer in enumerate(answers_data[:3]):
                                st.write(f"**A{i+1}:** {answer}")
                    
                    # Show validation summary
                    if questions_data and answers_data:
                        st.subheader("🔍 Extraction Validation")
                        
                        # Check if counts match
                        if len(questions_data) == len(answers_data):
                            st.success(f"✅ Perfect match: {len(questions_data)} questions and {len(answers_data)} answers")
                        else:
                            st.error(f"❌ Mismatch: {len(questions_data)} questions vs {len(answers_data)} answers")
                            st.warning("This mismatch will cause accuracy issues during testing!")
                        
                        # Show answer distribution
                        if answers_data:
                            from collections import Counter
                            answer_counts = Counter(answers_data)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("A answers", answer_counts.get('A', 0))
                            with col2:
                                st.metric("B answers", answer_counts.get('B', 0))
                            with col3:
                                st.metric("C answers", answer_counts.get('C', 0))
                            with col4:
                                st.metric("D answers", answer_counts.get('D', 0))
                            
                            # Check for distribution issues
                            total_answers = len(answers_data)
                            max_count = max(answer_counts.values()) if answer_counts else 0
                            max_percentage = (max_count / total_answers * 100) if total_answers > 0 else 0
                            
                            if max_percentage > 60:
                                st.warning(f"⚠️ One answer dominates ({max_percentage:.1f}%) - may indicate extraction error")
                            else:
                                st.success("✅ Answer distribution appears reasonable")
                    
                    # Save extracted data for debugging
                    processor.save_extracted_data(questions_data, answers_data)
                    logger.info("Extraction process completed successfully")
                    
                except ImportError as e:
                    logger.error(f"Missing module during extraction: {e}")
                    st.error(f"❌ Missing module: {str(e)}")
                    st.info("Please ensure test_processor.py is available in the scripts directory.")
                except Exception as e:
                    logger.error(f"Error during extraction: {e}")
                    st.error(f"❌ Error during extraction: {str(e)}")
                    st.info("Please check the PDF files and try again.")
                    # Clean up temporary files on error
                    try:
                        if 'test_path' in locals() and os.path.exists(test_path):
                            os.remove(test_path)
                        if 'answers_path' in locals() and os.path.exists(answers_path):
                            os.remove(answers_path)
                    except:
                        pass
    
    # Show extraction status
    if st.session_state.test_workflow_state['questions_extracted']:
        st.success("✅ Questions extracted successfully")
    if st.session_state.test_workflow_state['answers_extracted']:
        st.success("✅ Answers extracted successfully")
    
    # Step 2: Agent Configuration and Test Execution
    if st.session_state.test_workflow_state['step'] >= 2:
        st.markdown("---")
        st.subheader("⚙️ Step 2: Configure Agent and Run Test")
        
        # Agent configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_knowledge_base = st.checkbox("Use Knowledge Base", value=True, key="test_use_kb")
        with col2:
            use_web_search = st.checkbox("Use Web Search", value=True, key="test_use_web")
        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, key="test_temperature")
        
        # Display agent configuration summary
        st.info(f"**Agent Configuration:**\n"
                f"- Model: {os.getenv('AGENT_MODEL', 'gpt-3.5-turbo')}\n"
                f"- Knowledge Base: {'✅' if use_knowledge_base else '❌'}\n"
                f"- Web Search: {'✅' if use_web_search else '❌'}\n"
                f"- Temperature: {temperature}")
        
        # Step 2 Button: Run Automated Test
        test_button_disabled = not (st.session_state.test_workflow_state['questions_extracted'] and 
                                   st.session_state.test_workflow_state['answers_extracted'])
        
        if st.button("🚀 Run Automated Test", 
                     type="primary", 
                     use_container_width=True,
                     disabled=test_button_disabled,
                     key="run_automated_test"):
            
            logger.info("Starting automated test execution")
            
            # Check if base components are ready
            llm, system_prompt = initialize_base_agent()
            if not llm or not system_prompt:
                st.error("❌ Agent Failed to Initialize")
                logger.error("Agent failed to initialize for test execution")
                return
            
            # Create a single progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create progress callback for UI updates
            def progress_callback(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Progress: {current}/{total} - {message}")
                logger.info(f"Test progress: {current}/{total} - {message}")
            
            # Run automated test with single agent
            with st.spinner("🔄 Running automated test..."):
                try:
                    from scripts.test_runner import AutomatedTestRunner
                    
                    # Simplified agent config
                    agent_config = {
                        'model': os.getenv("AGENT_MODEL", "gpt-3.5-turbo"),
                        'temperature': temperature,
                        'timeout': 3600
                    }
                    
                    logger.info(f"Created simplified agent config: {agent_config}")
                    
                    runner = AutomatedTestRunner(agent_config)
                    
                    # Use cached data if available, otherwise run with extracted data
                    if skip_extraction:
                        logger.info("Running test with cached data")
                        results = runner.run_test_with_cached_data(progress_callback=progress_callback)
                    else:
                        logger.info("Running test with extracted data")
                        extraction_results = st.session_state.test_workflow_state['extraction_results']
                        # This will now use simplified processing with individual reasoning
                        results = runner.run_test_with_extracted_data(
                            extraction_results['questions'],
                            extraction_results['answers'],
                            progress_callback=progress_callback
                        )
                    
                    # Clean up temporary files after test completion
                    if not skip_extraction and st.session_state.test_workflow_state['extraction_results']:
                        try:
                            test_path = st.session_state.test_workflow_state['extraction_results']['test_path']
                            answers_path = st.session_state.test_workflow_state['extraction_results']['answers_path']
                            if os.path.exists(test_path):
                                os.remove(test_path)
                                logger.info(f"Cleaned up {test_path}")
                            if os.path.exists(answers_path):
                                os.remove(answers_path)
                                logger.info(f"Cleaned up {answers_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temporary files: {e}")
                    
                    # Check if results is None
                    if results is None:
                        st.error("❌ Test runner returned no results")
                        logger.error("Test runner returned None results")
                        return
                    
                    # Add defensive checks for required keys
                    required_keys = ['score_percentage', 'questions_answered', 'correct_answers']
                    missing_keys = [key for key in required_keys if key not in results]
                    
                    if missing_keys:
                        logger.error(f"Missing required keys in results: {missing_keys}")
                        logger.error(f"Available keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                        st.error(f"❌ Results missing required data: {missing_keys}")
                        return
                    
                    # Clear the progress bar and status text after completion
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mark test as completed
                    st.session_state.test_workflow_state['test_completed'] = True
                    
                    # Display results with safe access
                    score = results.get('score_percentage', 0)
                    st.success(f"🎉 Test completed! Score: {score:.1f}%")
                    logger.info(f"Test completed with score: {score:.1f}%")
                    
                    # Display detailed results
                    st.subheader("📊 Test Results Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Questions", results['questions_answered'])
                    with col2:
                        st.metric("Correct Answers", results['correct_answers'])
                    with col3:
                        st.metric("Score", f"{results['score_percentage']:.1f}%")
                    
                    # Generate and display report
                    from scripts.results_generator import ResultsGenerator
                    
                    # Add missing keys to results before generating report
                    if 'agent_config' not in results:
                        results['agent_config'] = {}
                    
                    if 'tools' not in results['agent_config']:
                        # Reconstruct tools info based on what was used
                        tools_used = []
                        if use_knowledge_base:
                            tools_used.append('knowledge_base_retriever')
                        if use_web_search:
                            tools_used.append('web_search_tool')
                        results['agent_config']['tools'] = tools_used
                    
                    # Ensure other required agent_config keys exist
                    results['agent_config']['model'] = results['agent_config'].get('model', os.getenv("AGENT_MODEL", "gpt-3.5-turbo"))
                    results['agent_config']['temperature'] = results['agent_config'].get('temperature', temperature)
                    
                    generator = ResultsGenerator(results)
                    report = generator.generate_comprehensive_report()
                    
                    with st.expander("📋 View Detailed Report"):
                        st.markdown(report)
                    
                    # Download links
                    st.download_button(
                        "📥 Download Full Report",
                        data=report.encode('utf-8'),
                        file_name=f"practice_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key="download_report"
                    )
                    
                    # Show JSON results as well
                    json_data = json.dumps(results, indent=2, default=str, ensure_ascii=False)
                    st.download_button(
                        "📥 Download Raw Results (JSON)",
                        data=json_data.encode('utf-8'),
                        file_name=f"practice_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_json"
                    )
                    
                except FileNotFoundError as e:
                    logger.error(f"Cached data not found: {e}")
                    st.error(f"❌ Cached data not found: {str(e)}")
                    st.info("Please run the test with PDFs first to generate cached data.")
                except Exception as e:
                    logger.error(f"Error running test: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Exception args: {e.args}")
                    
                    # Add stack trace for debugging
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    
                    # Check what we have in results if it exists
                    if 'results' in locals():
                        logger.error(f"Results variable exists: {results is not None}")
                        if results is not None:
                            logger.error(f"Results type: {type(results)}")
                            if isinstance(results, dict):
                                logger.error(f"Results keys: {list(results.keys())}")
                            else:
                                logger.error(f"Results content: {str(results)[:500]}")
                    else:
                        logger.error("Results variable does not exist in locals()")
                    
                    # Enhanced error display
                    st.error(f"❌ Error running test: {str(e)}")
                    st.error(f"Exception type: {type(e).__name__}")
                    
                    # Show more debugging info in UI
                    with st.expander("🐛 Debug Information"):
                        st.code(f"Exception: {e}")
                        st.code(f"Exception type: {type(e).__name__}")
                        st.code(f"Exception args: {e.args}")
                        if 'results' in locals() and results is not None:
                            if isinstance(results, dict):
                                st.code(f"Available result keys: {list(results.keys())}")
                            st.code(f"Results content: {str(results)[:1000]}")
                        st.code(traceback.format_exc())
                    
                    st.info("Please check the logs for more details.")
    
    # Step 3: Results and New Test Option
    if st.session_state.test_workflow_state.get('test_completed', False):
        st.markdown("---")
        st.subheader("🔄 Step 3: Start New Test")
        
        if st.button("🆕 Start New Test", key="start_new_test"):
            logger.info("User started new test, resetting workflow state")
            st.session_state.test_workflow_state = {
                'step': 1,
                'questions_extracted': False,
                'answers_extracted': False,
                'extraction_results': None,
                'test_completed': False
            }
            st.rerun()
    
    # Progress indicator in sidebar
    st.sidebar.markdown("### 📊 Progress")
    
    # Step 1 indicator
    if st.session_state.test_workflow_state['step'] >= 1:
        if st.session_state.test_workflow_state['questions_extracted'] and st.session_state.test_workflow_state['answers_extracted']:
            st.sidebar.success("✅ Step 1: Files Uploaded & Extracted")
        else:
            st.sidebar.info("⏳ Step 1: Upload Files & Extract")
    else:
        st.sidebar.info("⏳ Step 1: Upload Files & Extract")
    
    # Step 2 indicator
    if st.session_state.test_workflow_state['step'] >= 2:
        if st.session_state.test_workflow_state.get('test_completed', False):
            st.sidebar.success("✅ Step 2: Test Completed")
        else:
            st.sidebar.info("⏳ Step 2: Configure & Run Test")
    else:
        st.sidebar.info("⏳ Step 2: Configure & Run Test")
    
    # Step 3 indicator
    if st.session_state.test_workflow_state.get('test_completed', False):
        st.sidebar.success("✅ Step 3: Ready for New Test")
    else:
        st.sidebar.info("⏳ Step 3: New Test Option")

# Main function
def main():
    logger.info("Application main function started")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Initialized chat history in session state")
    
    # Navigation
    selected_page = st.sidebar.selectbox(
        "Select Page",
        ["Chat Interface", "Practice Test"],
        key="page_selector"
    )
    
    logger.info(f"User selected page: {selected_page}")
    
    # Route to appropriate interface
    if selected_page == "Chat Interface":
        chat_interface()
    elif selected_page == "Practice Test":
        practice_test_interface()

if __name__ == "__main__":
    logger.info("=== Starting Medical Coding AI Agent Application ===")
    main()
    logger.info("=== Application execution completed ===")