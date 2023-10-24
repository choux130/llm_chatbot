import boto3 
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def MaskStringWithAWSComprehend(raw_text: str) -> str:

    client = boto3.client('comprehend')
    response = client.detect_pii_entities(
        Text= raw_text,
        LanguageCode='en'
    )

    masked_text = raw_text
    for NER in reversed(response['Entities']):
        masked_text = masked_text[:NER['BeginOffset']] + \
        "***[" + raw_text[NER['BeginOffset']:NER['EndOffset']] + ' ---> ' + NER['Type'] + ']***' + \
        masked_text[NER['EndOffset']:]

    return masked_text

def ReadPDF(file_path) -> str:
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def SplitTextIntoChunk(text: str) -> list[str]:
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks

def AddVectorToFAISS(chunks: list[str]):
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def GetKnowledgeBase(text):

    chunks = SplitTextIntoChunk(text)
    knowledgeBase = AddVectorToFAISS(chunks)
    
    return knowledgeBase