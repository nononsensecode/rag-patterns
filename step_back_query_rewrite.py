from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

system_prompt = """You are an expert at taking a specific question and extracting a more generic question that gets at
the underlying principles needed to answer the original question. 

Given a question, write a more generic version of the question that needs to be answered to answer the original question.

If you don't recognize a word or acronym do not try to rewrite it.
Write concise questions."""

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

response = chain.invoke(
    {
        "question": "What is the best way to implement a neural network in Python using TensorFlow?"
    }
)
print(response)
