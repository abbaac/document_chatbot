import os
import json
from docx import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from cohere import Client

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = Client(COHERE_API_KEY)


with st.sidebar:
    st.title('Document Interaction Chatbot 📝')
    st.markdown('''
    ## About
    This chatbot allows you to upload a document and interact with its contents through a chat interface.

    ### Features:
    - Supports multiple document formats: `.txt`, `.json`, `.pdf`, `.docx`.
    - Ask questions based on the content of the uploaded document.
    - Get instant answers using AI-powered natural language processing.

    ### How to Use:
    1. Upload your document using the file uploader.
    2. Type your question about the document content in the chat box.
    3. Receive an AI-generated answer based on your document.

    ### Technologies:
    - Python
    - Natural Language Processing (NLP)
    - Cohere API for model prediction
    - Streamlit for the user interface

    **Explore your documents intelligently!**
    ''')

    add_vertical_space(2)
    st.write('Made by Abba Ali-Concern')


def read_txt(file):
    return file.read().decode('utf-8')


def read_json(file):
    return json.load(file)


def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(file_path, file_type):
    if file_type == 'txt':
        return read_txt(file_path)
    elif file_type == 'json':
        return read_json(file_path)
    elif file_type == 'pdf':
        return read_pdf(file_path)
    elif file_type == 'docx':
        return read_docx(file_path)
    else:
        return "Unsupported file format."


# Function to append conversation to chat history in session state
def update_chat_history(user_input, bot_response):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"user": user_input, "bot": bot_response})


def query_cohere(chat_history, user_input, document_text):
    # Append the document content at the start (as a reference)
    full_prompt = f"Document content: {document_text}\n\n"

    # Add chat history to the prompt
    for entry in chat_history:
        full_prompt += f"User: {entry['user']}\nChatbot: {entry['bot']}\n"

    # Add current user input
    full_prompt += f"User: {user_input}\nChatbot:"

    response = cohere_client.generate(
        model='command-xlarge-nightly',  # Choose model
        prompt=full_prompt,
        max_tokens=300,  # Limit response length
        temperature=0.7,  # Creativity level
        k=0,
        stop_sequences=["\n"],
        return_likelihoods='NONE'
    )

    # Return generated response
    return response.generations[0].text.strip()


def main():
    st.header("Hi there,")
    st.subheader("Get started by uploading your document 👇")
    add_vertical_space(2)
    uploaded_file = st.file_uploader("Upload your document", type=['txt', 'json', 'pdf', 'docx'])

    if uploaded_file is not None:

        # Extract file properties
        file_type = uploaded_file.name.split(".")[-1]
        file_name = uploaded_file.name.split(".")[0]

        # Convert file into raw text
        text = extract_text(uploaded_file, file_type)
        st.write("Document loaded successfully.")

        # Display previous conversation from session state
        if 'chat_history' in st.session_state and len(st.session_state.chat_history) > 0:
            st.subheader("Conversation History")
            for entry in st.session_state.chat_history:
                user_message = f"<div style='padding: 10px; background-color: #e0f7fa; border-radius: 10px; margin: 5px 0;'>" \
                               f"<strong>You:</strong> {entry['user']}</div>"
                bot_message = f"<div style='padding: 10px; background-color: #ffe0b2; border-radius: 10px; margin: 5px 0;'>" \
                              f"<strong>Chatbot:</strong> {entry['bot']}</div>"
                st.markdown(user_message, unsafe_allow_html=True)
                st.markdown(bot_message, unsafe_allow_html=True)

        add_vertical_space(2)
        # Ask chatbot questions
        user_query = st.text_input("Ask questions about your document:")

        # Chatbot Response
        if st.button("Get Answer") and user_query:
            # Get chat history from session state
            chat_history = st.session_state.chat_history if 'chat_history' in st.session_state else []

            with st.spinner("Generating response..."):
                # Query cohere
                response = query_cohere(chat_history, user_query, text)

                # Update chat history
                update_chat_history(user_query, response)

            # Better UI for chatbot response
            st.markdown("<h3>Chatbot Response:</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='border: 1px solid #d1d1d1; border-radius: 5px; padding: 10px; background-color: #f9f9f9;'>{response}</div>",
                unsafe_allow_html=True)


if __name__ == '__main__':
    main()



# if os.path.exists(f"{file_name}.pkl"):
#     with open(f"{file_name}.pkl", "rb") as f:
#         vector_store = pickle.load(f)
#     st.write("Embeddings loaded from the Disk")
# else:
#     # Generate embeddings for the text chunks
#     embeddings = cohere_client.embed(texts=chunks).embeddings
#
#     # Create a vector store
#     vector_store = {chunk: embedding for chunk, embedding in zip(chunks, embeddings)}
#     with open(f"{file_name}.pkl", "wb") as f:
#         pickle.dump(vector_store, f)
#     st.write("Embeddings created and saved to Disk")
#