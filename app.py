import streamlit as st
import pandas as pd
from wikipediaapi import Wikipedia
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from scipy.spatial.distance import cosine
import numpy as np

# Initialize Wikipedia API
wiki_wiki = Wikipedia(language='en', user_agent='saksath1016@gmail.com')

# Load chatbot model
CHAT_MODEL_NAME = "gemini-1.5-flash-001"
model = GenerativeModel(CHAT_MODEL_NAME)

# Load embeddings model
EMBED_MODEL_NAME = 'textembedding-gecko@003'
embedding_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)

# Function to get embeddings for texts
def embed_text(texts: list, model_name: str = EMBED_MODEL_NAME) -> list:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [np.array(embedding.values) for embedding in embeddings]
    except Exception as e:
        raise RuntimeError(f"Error embedding texts: {e}")

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - cosine(vec1, vec2)

# Function to fetch Wikipedia page content and save to CSV
def fetch_and_save_wikipedia_page(title):
    try:
        page = wiki_wiki.page(title)
        if page.exists():
            content = page.text
            paragraphs = content.split('\n\n')
            df = pd.DataFrame(paragraphs, columns=['paragraph'])
            df.to_csv("wikipedia_page.csv", index=False)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# Function to load data and embeddings
def load_data_and_embeddings():
    df = pd.read_csv("wikipedia_page.csv")
    paragraphs = df['paragraph'].tolist()
    embeddings = embed_text(paragraphs)
    return paragraphs, embeddings

# Function to find the most relevant paragraphs
def find_most_relevant_paragraphs(question, paragraphs, embeddings, top_n=5):
    question_embedding = np.array(embedding_model.get_embeddings([TextEmbeddingInput(question)])[0].values)
    similarities = [cosine_similarity(question_embedding, paragraph_embedding) for paragraph_embedding in embeddings]
    most_relevant_indices = np.argsort(similarities)[::-1][:top_n]
    return [paragraphs[i] for i in most_relevant_indices]

# Function to generate chatbot response
def generate_chatbot_response(question, context):
    chat = model.start_chat()
    bot_response = chat.send_message([context, question]).candidates[0].content.parts[0].text
    return bot_response

# Streamlit application
def main():
    st.title("Wikipedia Chatbot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input field for Wikipedia page title
    wikipedia_page_title = st.text_input("Enter Wikipedia Page Title:")

    # Button to fetch and process Wikipedia page
    if st.button("Fetch and Process Page"):
        if fetch_and_save_wikipedia_page(wikipedia_page_title):
            st.success("Page content saved to wikipedia_page.csv")
        else:
            st.error("Failed to fetch Wikipedia page. Please check the title.")

    # Button to load data and embeddings
    if st.button("Load Data and Embeddings"):
        paragraphs, embeddings = load_data_and_embeddings()
        st.session_state['paragraphs'] = paragraphs
        st.session_state['embeddings'] = embeddings
        st.success("Data and embeddings loaded successfully")

    # Accept user input
    user_question = st.text_input("What is your question?")

    # Respond to user input
    if user_question:
        if "paragraphs" in st.session_state and "embeddings" in st.session_state:
            paragraphs = st.session_state["paragraphs"]
            embeddings = st.session_state["embeddings"]

            if paragraphs and embeddings:
                most_relevant_paragraphs = find_most_relevant_paragraphs(user_question, paragraphs, embeddings)
                context = "\n\n".join(most_relevant_paragraphs)
                bot_response = generate_chatbot_response(user_question, context)

                # Add user question to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "bot", "content": bot_response})

                # Display user question and chatbot response
                st.markdown(f"**User:** {user_question}")
                with st.expander("Context"):
                    for i, paragraph in enumerate(most_relevant_paragraphs, start=1):
                        st.write(f"**Paragraph {i}:** {paragraph}")
                st.markdown(f"**Chatbot:** {bot_response}")

                # Display chat history
                st.subheader("Chat History")
                chat_history_text = ""
                for entry in st.session_state.chat_history:
                    role = "User" if entry["role"] == "user" else "Chatbot"
                    chat_history_text += f"**{role}:** {entry['content']}\n\n"
                st.markdown(chat_history_text)

        else:
            st.warning("Please load data and embeddings before asking a question.")

if __name__ == "__main__":
    main()