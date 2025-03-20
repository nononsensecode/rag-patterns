from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

system = """You are a helpful assistant that generates multiple search queries based on a single input query.

Perform query expansion. If there are multiple common ways of phrasing a user question
or common ways for key words in the question, make sure to return multiple versions of the
query with the different phrasings.

If there are acronyms or words you are not familiar with, do not rephrase them.
Return 3 versions of the question in a comma separated list."""

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
        "question": "Which food items does this recipe need?",
    }
)

print(response)
