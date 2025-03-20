from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

system = """You are a helpful assistant that generates multiple search queries based on a single input query.
Perform query decompositon. Given a user question, break it down into distinct sub-queries that
you need to answer in order to answer the original question.

If there are acronyms or words you are not familiar with, do not rephrase them.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)
chain = prompt | llm | StrOutputParser()

response = chain.invoke(
    {
        "question": """Which is the most popular programming language for machine learning and
        is it the most popular programming language overall?""",
    }
)
print(response)
