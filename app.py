# app.py

import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

 
st.set_page_config(
    page_title="Business Case Study Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Business & Finance Case Study Interviewer")
st.write("Hello! I am your expert interviewer. Ask me anything about your case studies.")



@st.cache_resource
def load_models_and_clients():
    """
    Load all the necessary models and clients for the RAG application.
    This function is cached to avoid reloading on every user interaction.
    """

    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    try:
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )


        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=gemini_api_key)

        llm = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        )
        return embedding_model, client, llm
    except Exception as e:
        st.error(f"Error loading models or clients: {e}")
        return None, None, None

embedding_model, client, llm = load_models_and_clients()


if embedding_model and client and llm:

    COLLECTION_NAME = "csot_data_consult"
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )
else:
    st.stop() 


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

qa_system_prompt = (
    "You are an expert interviewer for questioning and interviewing about business and finance case studies. "
    "Use the following pieces of retrieved context to ask and answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise and professional."
)



if query := st.chat_input("Ask your question here..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

 
    with st.spinner("Thinking..."):
        try:
     
            chat_history_for_prompt = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
            )
            
            contextualize_q_prompt = f"""{contextualize_q_system_prompt}

            Chat History:
            {chat_history_for_prompt}

            User Question: {query}"""

            response_1 = llm.generate_content(contextualize_q_prompt)
            standalone_question = response_1.text.strip()
            
       
            with st.expander("Standalone Question"):
                st.write(standalone_question)

            
            retrieved_docs = vector_store.similarity_search(query=standalone_question, k=5)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            
            qa_prompt = f"""{qa_system_prompt}

            Context:
            {context}

            Chat History:
            {chat_history_for_prompt}

            Question: {query}
            """
            response_2 = llm.generate_content(qa_prompt)
            final_answer = response_2.text.strip()

      
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
