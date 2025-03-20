from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

doc_url = "https://arxiv.org/pdf/2312.10997.pdf"
loader = PyPDFLoader(doc_url)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(pages)
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
chunk_texts = list(map(lambda x: x.page_content, chunks))
embeddings = bge_embeddings.embed_documents(chunk_texts)
text_embedding_pairs = zip(chunk_texts, embeddings)
db = FAISS.from_embeddings(text_embedding_pairs, bge_embeddings)

query = "Which are the drawbacks of Naive RAG?"
contexts = db.similarity_search(query, k=5)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in answering questions based on a context received from a document. The context is {context}",
        ),
        (
            "human",
            "Based on the context, please answer the following question: {question}",
        ),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"context": "\n\n".join(chunk_texts), "question": query})
print(response)
