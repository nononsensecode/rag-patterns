from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

actual_document = """Berkson's paradox, also known as Berkson's bias, collider bias or Berkson's fallacy is a result in conditional probability
and statistics which is often found to be counterintuitive, and hence a veridical paradox. It is a complicating factor arising in
statistical tests of proportions. Specifically, it is arises when there is an ascertainment bias inherent in a study design. The effect is
related to the explaining away phenomenon in Bayesian networks, and conditioning on a collider in graphical models.

It is often described in the fields of medical statistics or biostatistics as in the original description of the problem by Joseph Berkson
"""
system_prompt = """You are an expert at using a question to generate a document useful for answering the question.
Given a question generate a paragraph of text that answers the question. 
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

chain = prompt | llm | StrOutputParser()
hypothetical_response = chain.invoke(
    {
        "question": "What does Berkson's paradox consist on?",
    }
)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
chunk_texts = [actual_document]
actual_embeddings = bge_embeddings.embed_documents(chunk_texts)
question_embedding = bge_embeddings.embed_documents(
    ["What does Berkson's paradox consist on?"]
)
hypothetical_embeddings = bge_embeddings.embed_documents(
    [hypothetical_response],
)

print(
    f"Similarity without HyDE: {cosine_similarity(actual_embeddings, question_embedding)}"
)
print(
    f"Similarity with HyDE: {cosine_similarity(hypothetical_embeddings, question_embedding)}"
)
