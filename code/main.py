import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from helper import * 

# Load environment variables
load_dotenv()

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.title("")
    st.sidebar.title('Talk to  your PDF')
    pdf_file = st.sidebar.file_uploader('Upload your PDF Document', type='pdf')

    tab1, tab2 = st.tabs(["Chatbot", "Extracted Text with PII masked"])

    if pdf_file is not None:
        try:
            text = ReadPDF(pdf_file)
            # st.sidebar.info("The content of the PDF is hidden. Type your query in the chat window.")
            
            masked_text = MaskStringWithAWSComprehend(text)
            # st.sidebar.text_area('The following are the text read from the uploaded pdf:', masked_text, height=200)
            tab2.subheader('Text read from the uploaded pdf with PII redaction flagged with this format, **[raw_string --> entity_name]**:')
            tab2.write([masked_text])

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
            
            knowledgeBase = GetKnowledgeBase(text)

            docs = knowledgeBase.similarity_search(prompt)
            llm = OpenAI(streaming = True)
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=prompt)
                print(cost)
            
            message_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()