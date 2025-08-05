import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")

# Initialize HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

# Optional: Inject API token into client manually if needed
llm.client._token = api_key

# Initialize Chat Model
chat_model = ChatHuggingFace(llm=llm)

# Streamlit App Title
st.title("LLM Chatbot - Streamlit + LangChain + HuggingFace")

# Initialize Session State for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User Input Box
user_input = st.chat_input("Type your message...")

# On user submit
if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke model
    with st.spinner("Thinking..."):
        result = chat_model.invoke(st.session_state.chat_history)
        ai_response = result.content

    # Add AI response to history
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)
