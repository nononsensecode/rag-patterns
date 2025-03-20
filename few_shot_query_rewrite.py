from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

system = """You are a helpful assistant that generates multiple search queries based on a single input query.

Perform query expansion. If there are multiple common ways of phrasing a user question
or common ways for key words in the question, make sure to return multiple versions of the
query with the different phrasings.

If there are acronyms or words you are not familiar with, do not rephrase them.
Return 3 versions of the question"""

examples = [
    {
        "question": "What is the capital of France?",
        "answer": "1. What is the capital city of France? \n2.What is the capital of France?\n 3. What is the capital of France?",
    },
    {
        "question": "How to bake a cake?",
        "answer": "1. How do I bake a cake?\n2. What are the steps to bake a cake?\n3. How can I make a cake?",
    },
    {
        "question": "What is the weather like today?",
        "answer": "1. How is the weather today?\n2. What is today's weather?\n3. What will the weather be like today?",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{question}"), ("ai", "{answer}")]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        few_shot_prompt,
        ("human", "{question}"),
    ]
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)
chain = final_prompt | llm | StrOutputParser()

response = chain.invoke(
    {
        "question": "Which food items does this recipe need?",
    }
)
print(response)
