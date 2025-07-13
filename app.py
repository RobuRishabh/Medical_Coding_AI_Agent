import streamlit as st
import os
import json
import logging
import re
from dotenv import load_dotenv
from smolagents import CodeAgent  # Better choice than ToolCallingAgent
from scripts.smolagent_tools import knowledge_base_retriever, web_search_tool, TOOL_NAMES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datetime import datetime

# ───────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="CPC Medical Coding Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# LOGGING SETUP
# ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# LOAD ENV VARIABLES
# ───────────────────────────────────────────────
load_dotenv()

# ───────────────────────────────────────────────
# Load system prompt
# ───────────────────────────────────────────────
@st.cache_data
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

# ───────────────────────────────────────────────
# Safe wrapper for ChatGoogleGenerativeAI output
# ───────────────────────────────────────────────
class SafeLLMWrapper:
    def __init__(self, llm):
        self.llm = llm

    def convert_messages(self, messages):
        # Handle single message case
        if not isinstance(messages, list):
            messages = [messages]
            
        converted = []
        for msg in messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

            if role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            elif role == "system":
                converted.append(SystemMessage(content=content))
            elif hasattr(role, 'value') and role.value == 'tool-response':
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', str(content))
                else:
                    text_content = str(content)
                converted.append(HumanMessage(content=text_content))
            else:
                logger.warning(f"Unsupported role: {role}, converting to HumanMessage")
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', str(content))
                else:
                    text_content = str(content)
                converted.append(HumanMessage(content=text_content))
        return converted

    def generate(self, messages, **kwargs):
        try:
            converted_messages = self.convert_messages(messages)
            response = self.llm.invoke(converted_messages, **kwargs)
            
            # Post-process response to fix code block formatting
            content = getattr(response, "content", str(response))
            
            # Convert markdown code blocks to <code> tags expected by CodeAgent
            # Replace ```python or ```tool_code blocks with <code> tags
            content = re.sub(r'```(?:python|tool_code)?\s*\n(.*?)\n```', r'<code>\1</code>', content, flags=re.DOTALL)
            
            # Also handle single backtick code blocks
            content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
            
            # Create a modified response object
            class ModifiedResponse:
                def __init__(self, original_response, modified_content):
                    self.content = modified_content
                    # Copy other attributes from original response safely
                    for attr in dir(original_response):
                        if not attr.startswith('_') and attr != 'content':
                            try:
                                # Skip problematic Pydantic attributes
                                if attr not in ['model_computed_fields', 'model_fields', 'model_config']:
                                    setattr(self, attr, getattr(original_response, attr))
                            except:
                                pass
            
            modified_response = ModifiedResponse(response, content)
            
            class Result:
                def __init__(self, content, response_obj):
                    self.content = content
                    # Add tool_calls attribute for ToolCallingAgent
                    self.tool_calls = getattr(response_obj, 'tool_calls', None)
                    usage_data = getattr(response_obj, 'usage_metadata', {})
                    if isinstance(usage_data, dict):
                        class TokenUsage:
                            def __init__(self, data):
                                self.input_tokens = data.get('input_tokens', 0)
                                self.output_tokens = data.get('output_tokens', 0) 
                                self.total_tokens = data.get('total_tokens', 0)
                        self.token_usage = TokenUsage(usage_data)
                    else:
                        class TokenUsage:
                            def __init__(self):
                                self.input_tokens = 0
                                self.output_tokens = 0
                                self.total_tokens = 0
                        self.token_usage = TokenUsage()
            
            return Result(content=content, response_obj=modified_response)
        except Exception as e:
            logger.error(f"LLM.generate failed: {e}")
            raise

    def __getattr__(self, name):
        return getattr(self.llm, name)

# ───────────────────────────────────────────────
# Initialize Model
# ───────────────────────────────────────────────
@st.cache_resource
def initialize_model():
    logger.info("Initializing Google Gemini language model...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        raw_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        logger.info("Google Gemini model initialized successfully")
        return SafeLLMWrapper(raw_llm)
    except Exception as e:
        logger.error(f"Failed to initialize Google Gemini model: {e}")
        st.error(f"Failed to initialize model: {e}")
        return None

# ───────────────────────────────────────────────
# Initialize Agent with dynamic tools
# ───────────────────────────────────────────────
@st.cache_resource
def initialize_base_agent():
    llm = initialize_model()
    if not llm:
        return None, None
    
    try:
        logger.info("Creating base agent components...")
        system_prompt = load_system_prompt()
        
        # Debug: Check available tools
        logger.info(f"knowledge_base_retriever: {knowledge_base_retriever}")
        logger.info(f"web_search_tool: {web_search_tool}")
        
        # Log tool names if they have a name attribute
        if hasattr(knowledge_base_retriever, 'name'):
            logger.info(f"Knowledge base tool name: {knowledge_base_retriever.name}")
        if hasattr(web_search_tool, 'name'):
            logger.info(f"Web search tool name: {web_search_tool.name}")
        
        logger.info("Base agent components created successfully")
        return llm, system_prompt
    except Exception as e:
        logger.error(f"Failed to create base agent: {e}")
        st.error(f"Failed to create base agent: {e}")
        return None, None

def create_dynamic_agent(use_knowledge_base=True, use_web_search=True):
    """Create agent with selected tools"""
    llm, system_prompt = initialize_base_agent()
    if not llm:
        return None
    
    try:
        # Select tools based on user preferences
        selected_tools = []
        if use_web_search:  # Always include web search
            selected_tools.append(web_search_tool)
        if use_knowledge_base:
            selected_tools.append(knowledge_base_retriever)

        logger.info(f"Creating CodeAgent with {len(selected_tools)} tools...")
        
        # Log the actual tool names for debugging
        for tool in selected_tools:
            tool_name = getattr(tool, 'name', str(tool))
            logger.info(f"Available tool: {tool_name}")
        
        agent = CodeAgent(
            tools=selected_tools,
            model=llm,
            max_steps=5, 
            additional_authorized_imports=["re", "json", "os"]  # Add authorized imports
        )
        
        # Get actual tool names from the selected tools
        tool_names = [tool.name for tool in selected_tools]
        
        logger.info(f"Actual tool names: {tool_names}")
        
        # Enhanced system prompt with correct tool names
        enhanced_system_prompt = system_prompt.format(tool_names=tool_names)
        
        # Set enhanced system prompt
        agent.prompt_templates["system_prompt"] = enhanced_system_prompt
        
        logger.info("Dynamic agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create dynamic agent: {e}")
        st.error(f"Failed to create dynamic agent: {e}")
        return None

def extract_citations_from_response(response_text, agent):
    """Extract and format citations from agent response"""
    citations = {
        'web_sources': [],
        'knowledge_base_sources': []
    }
    
    # Try to extract citations from the response text
    if "**Sources:**" in response_text:
        # Parse existing citations
        sources_section = response_text.split("**Sources:**")[1] if "**Sources:**" in response_text else ""
        
        # Extract web sources
        web_pattern = r'- Web Search: \[([^\]]+)\]\(([^)]+)\)(.*)'
        web_matches = re.findall(web_pattern, sources_section)
        for match in web_matches:
            citations['web_sources'].append({
                'name': match[0],
                'url': match[1],
                'description': match[2].strip(' -')
            })
        
        # Extract knowledge base sources
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
    """Display which tools were used"""
    st.markdown("---")
    st.subheader("🔧 Tools Used")
    
    # Tool usage summary
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

def chat_interface():
    """Chat interface page"""
    st.title("🩺 CPC Medical Coding Assistant - Chat Interface")
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Your Question")
        
        # Initialize session state for question if not exists
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # Input area
        question = st.text_area(
            "Enter your medical coding question:",
            placeholder="e.g., What is the CPT code for a routine colonoscopy?",
            height=100,
            value=st.session_state.current_question
        )
        
        # Update session state when text area changes
        if question != st.session_state.current_question:
            st.session_state.current_question = question
        
        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "How do I code a chest X-ray with interpretation?",
            "What ICD-10 code should I use for Type 2 diabetes?",
            "What's the difference between CPT 99213 and 99214?",
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📝 {example}", key=f"example_{i}"):
                st.session_state.current_question = example
                st.rerun()

    with col2:
        st.header("⚙️ Options")
        # Web search is always enabled (can't be turned off)
        st.markdown("🌐 **Web Search**: Always Enabled")
        st.info("Web search is always active to provide the most current information.")
        use_web_search = True  # Always True
        
        # Search options
        use_knowledge_base = st.checkbox("🔍 Search Knowledge Base", value=True)
        st.info("Knowledge base search is optional. Uncheck to skip searching the embedded documents.")
        # Display options
        show_sources = st.checkbox("📖 Show Sources", value=True)

    # Submit button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not st.session_state.current_question.strip():
            st.warning("Please enter a question!")
            return
        
        # Check if base components are ready
        llm, system_prompt = initialize_base_agent()
        if not llm or not system_prompt:
            st.error("❌ Agent Failed to Initialize")
            return
        
        # Show processing
        with st.spinner("🔄 Processing your question..."):
            try:
                logger.info(f"Processing query: {st.session_state.current_question}")
                logger.info(f"Using tools - Knowledge Base: {use_knowledge_base}, Web Search: {use_web_search}")
                
                # Create agent with selected tools
                agent = create_dynamic_agent(use_knowledge_base, use_web_search)
                if not agent:
                    st.error("Failed to create agent with selected tools")
                    return
                
                # Get response from agent
                response = agent.run(st.session_state.current_question)
                
                # Display results
                st.markdown("---")
                st.header("🤖 Answer")
                
                # Format and display response
                if isinstance(response, str):
                    st.markdown(response)
                    response_text = response
                else:
                    st.write(response)
                    response_text = str(response)
                
                # Show enhanced sources and references if requested
                if show_sources:
                    display_enhanced_sources(response_text, use_knowledge_base, use_web_search)
                
                logger.info("Query processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"⚠️ Error: {str(e)}")
                st.info("Please try rephrasing your question or check the system logs.")

def practice_test_interface():
    """Interface for automated practice test"""
    st.title("🩺 CPC Medical Coding Assistant - Practice Test")
    st.markdown("---")
    
    st.header("🎯 Automated Practice Test Runner")
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        test_file = st.file_uploader(
            "Upload Practice Test (PDF)", 
            type=['pdf'],
            help="Upload the practice test without answers"
        )
    
    with col2:
        answers_file = st.file_uploader(
            "Upload Answer Key (PDF)", 
            type=['pdf'],
            help="Upload the practice test with answers"
        )
    
    # Agent configuration
    st.subheader("🔧 Agent Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_knowledge_base = st.checkbox("Use Knowledge Base", value=True)
    with col2:
        use_web_search = st.checkbox("Use Web Search", value=True)
    with col3:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    
    # Run test button
    if st.button("🚀 Run Automated Test", type="primary"):
        if test_file and answers_file:
            # Check if base components are ready
            llm, system_prompt = initialize_base_agent()
            if not llm or not system_prompt:
                st.error("❌ Agent Failed to Initialize")
                return
            
            # Save uploaded files
            test_path = f"temp_test_{datetime.now().timestamp()}.pdf"
            answers_path = f"temp_answers_{datetime.now().timestamp()}.pdf"
            
            with open(test_path, "wb") as f:
                f.write(test_file.read())
            with open(answers_path, "wb") as f:
                f.write(answers_file.read())
            
            # Run automated test
            with st.spinner("Running automated test..."):
                # Updated import path
                from scripts.test_runner import AutomatedTestRunner
                
                agent_config = {
                    'model': 'gemini-2.0-flash',
                    'tools': ['knowledge_base_retriever', 'web_search'] if use_knowledge_base and use_web_search else ['web_search'],
                    'temperature': temperature
                }
                
                runner = AutomatedTestRunner(agent_config)
                results = runner.run_complete_test(test_path, answers_path)
                
                # Display results
                st.success(f"Test completed! Score: {results['score_percentage']:.1f}%")
                
                # Generate and display report
                # Updated import path
                from scripts.results_generator import ResultsGenerator
                generator = ResultsGenerator(results)
                report = generator.generate_comprehensive_report()
                
                st.markdown(report)
                
                # Download links
                st.download_button(
                    "📥 Download Full Report",
                    data=report,
                    file_name=f"practice_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        else:
            st.error("Please upload both test file and answer key")

# ───────────────────────────────────────────────
# MAIN APP WITH NAVIGATION
# ───────────────────────────────────────────────
def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Navigation
    selected_page = st.sidebar.selectbox(
        "Select Page",
        ["Chat Interface", "Practice Test"]
    )
    
    if selected_page == "Chat Interface":
        chat_interface()
    elif selected_page == "Practice Test":
        practice_test_interface()

if __name__ == "__main__":
    main()