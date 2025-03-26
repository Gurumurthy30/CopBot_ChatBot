import streamlit as st
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import pickle

# Global variables to store loaded components
_embedding_model = None
_bm25_retriever = None
_faiss_store = None
_ensemble_retriever = None
_llm = None
_retrieval_chain = None
_components_loaded = False
_memory = None

def initialize():
    """Initialize all components once at startup"""
    global _embedding_model, _bm25_retriever, _faiss_store, _ensemble_retriever, _llm, _retrieval_chain, _components_loaded, _memory
    
    # Check if components are already loaded
    if _components_loaded:
        return
    
    with st.spinner("Loading components... This may take a minute..."):
        start_time = time.time()
        
        # Load environment variables
        load_dotenv()
        os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
        os.environ['GROQ_API_KEY'] = os.getenv("GROQ")
        
        # Initialize LLM
        _llm = ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0.3
        )
        
        # Initialize memory with explicit output_key
        _memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load embeddings model
        _embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load BM25 retriever
        bm25_path = r"C:/Users/gurum/Documents/police/ChatBot/saved_components/bm25_retriever.pkl"
        with open(bm25_path, "rb") as f:
            _bm25_retriever = pickle.load(f)
        
        # Load FAISS index safely
        faiss_path = r"C:\Users\gurum\Documents\police\ChatBot\saved_components\faiss_index"
        _faiss_store = FAISS.load_local(
            faiss_path,
            _embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Create Ensemble Retriever
        _ensemble_retriever = EnsembleRetriever(
            retrievers=[_bm25_retriever, _faiss_store.as_retriever()],
            weights=[0.5, 0.5]
        )
        
        # Define the CopBot prompt with chat history
        template = """You are **CopBot**, an AI-powered police assistance chatbot that provides **official, verified, and retrieval-based information** about police procedures, complaint filing, emergency contacts, and legal provisions.  

### **Previous Conversation:**
{chat_history}

### **Instructions:**  
- **Multi-Language Support:** Automatically detect the user's language. If not English, translate to English for retrieval and then back to the original language for the response.  
- **Strict Retrieval-Based Responses:** Only use structured police records (Word documents, Google Sheets, Excel files). No assumptions or external sources.  
- **Direct & Concise Answers:** Provide only the final response. Do **not** include translation steps, retrieval process, or explanations.  
- **Error Handling:** If no relevant information is found, respond with:  
  *"I'm sorry, but I couldn't find official records on this. Please consult the nearest police station for further assistance."*  
- **Offline Mode Support:** Ensure seamless functionality without an internet connection when deployed in police stations or government offices.  
- **Remember Previous Interactions:** Use the chat history to provide more context-aware responses.

### **Response Flow:**  
1. Detect **user's input language** automatically.  
2. Translate query into **English** for document retrieval (if necessary).  
3. Retrieve the most **relevant official record**.  
4. Translate the **answer back to the original input language**.  
5. **Return only the final answer**—no intermediate steps, no extra explanations.  

---

**User Query:** {input}  
**Retrieved Context:** {context}  
**Final Answer (in user's original language):**"""
        
        # Set up prompt template with memory
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the document chain with string output parser
        document_chain = create_stuff_documents_chain(
            _llm, 
            prompt
        )
        
        # Create the modern retrieval chain
        _retrieval_chain = create_retrieval_chain(
            _ensemble_retriever,
            document_chain
        )
        
        _components_loaded = True
        end_time = time.time()
        st.success(f"Components loaded successfully in {end_time - start_time:.2f} seconds")

def query_copbot(query_text):
    """Process a query using the initialized components"""
    global _retrieval_chain, _components_loaded, _memory
    
    # Ensure components are loaded
    if not _components_loaded:
        initialize()
    
    # Get chat history from memory
    chat_history = _memory.chat_memory.messages if _memory and hasattr(_memory, 'chat_memory') else []
    
    # Process the query and measure time
    start_time = time.time()
    result = _retrieval_chain.invoke({
        "input": query_text,
        "chat_history": chat_history
    })
    end_time = time.time()
    
    # Save to memory
    if _memory:
        _memory.chat_memory.add_user_message(query_text)
        _memory.chat_memory.add_ai_message(result["answer"])
    
    # Return result with timing info
    processing_time = end_time - start_time
    return result["answer"], processing_time, result.get("context", [])

# Streamlit UI
def main():
    st.set_page_config(
        page_title="CopBot - Police Assistant",
        page_icon="👮",
        layout="wide"
    )
    
    # Title and description
    st.title("👮 CopBot - Police Assistant")
    st.markdown("""
    A retrieval-based chatbot for official police information, procedures, and contacts.
    Ask questions about police procedures, complaint filing, emergency contacts, and legal provisions.
    """)
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    initialize()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**\n```\n{source.page_content[:500]}...\n```")
    
    # Chat input
    if query := st.chat_input("Ask CopBot a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Query the bot
            try:
                response, processing_time, sources = query_copbot(query)
                
                # Display response
                message_placeholder.markdown(response)
                
                # Show processing time in small text
                st.caption(f"Response generated in {processing_time:.2f} seconds")
                
                # Optionally show sources if available
                if sources:
                    with st.expander("View sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**\n```\n{source.page_content[:500]}...\n```")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Add a sidebar with additional information
    with st.sidebar:
        st.header("About CopBot")
        st.markdown("""
        CopBot is designed to provide verified information from official police records.
        
        **Features:**
        - Multi-language support
        - Retrieval-based responses
        - Conversation memory
        - Offline mode support
        
        **Sources:**
        CopBot uses information from authorized police documents only.
        """)
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            # Also clear the conversation memory
            global _memory
            if _memory and hasattr(_memory, 'chat_memory'):
                _memory.chat_memory.clear()
            st.rerun()

if __name__ == "__main__":
    main()