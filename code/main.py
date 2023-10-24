import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


# openai.api_key = st.secrets["OPENAI_API_KEY"]
# Load environment variables
load_dotenv()

def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.title("")
    st.sidebar.title('Talk to  your PDF')
    pdf_file = st.sidebar.file_uploader('Upload your PDF Document', type='pdf')

    if pdf_file is not None:
        try:
            text = read_pdf(pdf_file)
            st.sidebar.info("The content of the PDF is hidden. Type your query in the chat window.")
        except FileNotFoundError:
            st.error(f"File not found: {pdf_file}")
            return
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

    # We used the := operator to assign the user's input to the prompt variable and checked if it's not None in the same line. 
    # If the user has sent a message, we display the message in the chat message container and append it to the chat history.
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            full_response = ""
            knowledgeBase = process_text(text)
            docs = knowledgeBase.similarity_search(prompt)
            llm = OpenAI(streaming = True)
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=prompt)
                print(cost)
            full_response = response
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()