import streamlit as st
import os
import json
import logging
from dotenv import load_dotenv
from smolagents import ToolCallingAgent  # Better choice than CodeAgent
from smolagent_tools import knowledge_base_retriever, web_search_tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
            
            return Result(content=getattr(response, "content", str(response)), response_obj=response)
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
# Initialize Agent
# ───────────────────────────────────────────────
@st.cache_resource
def initialize_agent():
    llm = initialize_model()
    if not llm:
        return None
    
    try:
        logger.info("Creating ToolCallingAgent with retriever tools...")
        agent = ToolCallingAgent(
            tools=[knowledge_base_retriever, web_search_tool],
            model=llm,
            max_steps=3
        )
        
        # Set system prompt
        system_prompt = load_system_prompt()
        agent.prompt_templates["system_prompt"] = system_prompt
        
        logger.info("Agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        st.error(f"Failed to create agent: {e}")
        return None

# ───────────────────────────────────────────────
# MAIN STREAMLIT APP
# ───────────────────────────────────────────────
def main():
    # Header
    st.title("🩺 CPC Medical Coding Assistant")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This AI assistant helps with medical coding questions including:
        - **CPT Codes**: Procedures and services
        - **ICD-10**: Diagnoses and conditions  
        - **HCPCS**: Healthcare procedures
        - **Coding Guidelines**: Best practices
        - **Billing Questions**: Reimbursement help
        """)
        
        st.header("🔧 Status")
        # Initialize agent
        agent = initialize_agent()
        if agent:
            st.success("✅ Agent Ready")
        else:
            st.error("❌ Agent Failed to Initialize")
            st.stop()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Input area
        question = st.text_area(
            "Enter your medical coding question:",
            placeholder="e.g., What is the CPT code for a routine colonoscopy?",
            height=100
        )
        
        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "What is the CPT code for a routine colonoscopy?",
            "How do I code a chest X-ray with interpretation?",
            "What ICD-10 code should I use for Type 2 diabetes?",
            "What's the difference between CPT 99213 and 99214?",
            "How do I code a telehealth visit?"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📝 {example}", key=f"example_{i}"):
                question = example
                st.rerun()
    
    with col2:
        st.header("⚙️ Options")
        
        # Search options
        use_knowledge_base = st.checkbox("🔍 Search Knowledge Base", value=True)
        use_web_search = st.checkbox("🌐 Search Web", value=False)
        
        # Display options
        show_sources = st.checkbox("📖 Show Sources", value=True)
        
        st.markdown("---")
        st.markdown("**💡 Tip:** Use specific medical terms for better results")
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question!")
            return
        
        # Show processing
        with st.spinner("🔄 Processing your question..."):
            try:
                logger.info(f"Processing query: {question}")
                
                # Get response from agent
                response = agent.run(question)
                
                # Display results
                st.markdown("---")
                st.header("🤖 Answer")
                
                # Format and display response
                if isinstance(response, str):
                    st.markdown(response)
                else:
                    st.write(response)
                
                # Show additional info if requested
                if show_sources:
                    st.markdown("---")
                    st.subheader("📚 Sources")
                    st.info("Sources: Internal knowledge base and web search results")
                
                logger.info("Query processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"⚠️ Error: {str(e)}")
                st.info("Please try rephrasing your question or check the system logs.")

# ───────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    main()