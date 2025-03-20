from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

system_rewrite = """You are a helpful assistant that generates multiple search queries based on a single input query.

Perform query expansion. If there are multiple common ways of phrasing a user question
or common synonyms for key words in the question, make sure to return multiple versions
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Return 3 different versions of the question."""

document_url = "https://arxiv.org/pdf/2312.10997.pdf"
loader = PyPDFLoader(document_url)
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
chunk_texts = list(map(lambda d: d.page_content, chunks))
embeddings = bge_embeddings.embed_documents(chunk_texts)
text_embedding_pairs = zip(chunk_texts, embeddings)
db = FAISS.from_embeddings(text_embedding_pairs, bge_embeddings)
query = "Which evaluation tools are useful for evaluating a RAG pipeline?"


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        description="A unique paraphrasing of the original question."
    )


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "{question}",
        ),
    ]
)
llm_with_tools = model.bind_tools([ParaphrasedQuery])
query_analyzer = (
    rewrite_prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])
)
queries = query_analyzer.invoke(
    {
        "question": query,
    }
)
contexts = []

for query in queries:
    print(f"Query: {query.paraphrased_query}")
    contexts = contexts + db.similarity_search(query.paraphrased_query, k=1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at answering questions based on a context extracted from a document. The context extracted from the document is: {context}",
        ),
        ("human", "{question}"),
    ]
)

chain = prompt | model | StrOutputParser()
response = chain.invoke(
    {
        "context": "\n\n".join(list(map(lambda c: c.page_content, contexts))),
        "question": query,
    }
)
print(response)
