from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys_prompt = """You are an expert in summarizing text. Write a concisesummary of the text provided. 
The summary should be a list of bullet points. The text to summarize is {text}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        ("user", "{text}"),
    ]
)

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-flash-8b",
)


def summarize_text(splits: list[str]) -> str:
    """
    Summarize the text using Google Generative AI.

    Args:
        splits (list[str]): List of text chunks to summarize.

    Returns:
        str: The summary of the text.
    """
    # Create a list to store the summaries
    summaries = []

    chain = prompt | llm | StrOutputParser()

    # Iterate over each text chunk and summarize it
    for text in splits:
        # Generate the summary using the LLM
        summary = chain.invoke({"text": text})
        summaries.append(summary)

    # Join the summaries into a single string
    return "\n".join(summaries)
