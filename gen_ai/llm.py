from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
from colorama import Fore
from tqdm import tqdm  
from langchain_community.llms import Ollama

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI


embeddings = OllamaEmbeddings()

'''
api_key = "your_openai_api_key"
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
'''

INDEX_PATH = "faiss_index"

def load_or_create_vector_db():
    try:
        if os.path.exists(INDEX_PATH):
            print(Fore.YELLOW + "Loading FAISS index from local storage...")
            db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            print(Fore.YELLOW + "Local FAISS index not found. Creating new vector store...")

            loader = CSVLoader(file_path="gen/alumni_data.csv")
            transcript = loader.load()

            print(Fore.YELLOW + "Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

            docs = []
            for doc in tqdm(transcript, desc="Processing documents"):
                docs.extend(text_splitter.split_documents([doc]))

            print(Fore.YELLOW + "Creating FAISS vector store...")
            db = FAISS.from_documents(docs, embeddings)

            print(Fore.YELLOW + "Saving new FAISS index to disk...")
            db.save_local(INDEX_PATH)

        return db

    except Exception as e:
        raise RuntimeError(f"Failed to create or load vector DB: {str(e)}")

def get_response(db, query, k=8):
    try:
        print(Fore.YELLOW + "Performing similarity search...")
        docs = db.similarity_search(query, k=k)

        docs_page_content = ""
        for d in tqdm(docs, desc="Processing search results"):
            docs_page_content += d.page_content + " "

        llm = Ollama(model='llama2')
        #llm = OpenAI(model='gpt-4', openai_api_key=api_key)

        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template = """ You are a college alumni assistant. 
            Given the following extracted content from our database: {docs},
            respond concisely to the following question: {question}. 
            Please provide specific and factual information. """
        )

        print(Fore.YELLOW + "Running the LLM chain...")
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        return response

    except Exception as e:
        raise RuntimeError(f"Failed to get response: {str(e)}")

if __name__ == "__main__":
    db = load_or_create_vector_db()
    qn=input()
    response = get_response(db, qn)
    print(Fore.WHITE + response)
