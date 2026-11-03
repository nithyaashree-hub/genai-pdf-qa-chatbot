## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To develop a PDF-based question-answering chatbot using LangChain that can read and understand a PDF document, retrieve relevant information, and provide accurate answers to user questions. The chatbot is tested with different questions to check its performance.

### DESIGN STEPS:

#### STEP 1: 
Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: 
Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: 
Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: 
Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: 
Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.

### PROGRAM:
~~~
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

pdf_path = "sample_data_science_document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
print("NITHYAA SHREE T\n212225220069")

print("PDF Question Answering Chatbot")

while True:
    question = input("\nEnter your question (or type exit): ")

    if question.lower() == "exit":
        print("Thank you for using the chatbot!")
        break

    result = qa_chain.invoke({"query": question})

    print("\nAnswer:", result["result"])
~~~
### OUTPUT:

<img width="1920" height="1080" alt="Screenshot (622)" src="https://github.com/user-attachments/assets/0dd3566d-5d08-43da-bed2-a898394e41cd" />

### RESULT:

Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
