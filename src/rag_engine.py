import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            max_tokens=500
        )
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a Senior Customer Support AI for 'TechGadget Inc.'. 
        Your goal is to provide accurate, helpful answers based ONLY on the provided company policies.
        <Instructions>
        1. Read the Context carefully.
        2. If the answer is found in the Context, answer clearly and concisely.
        3. Cite the specific policy section if possible (e.g., "According to the Refund Policy...").
        4. If the answer is NOT in the Context, strictly state: "I cannot answer this based on the current company policies." Do not make up answers.
        5. Use bullet points for lists to improve readability.
        </Instructions>
        <Context>
        {context}
        </Context>
        <Question>
        {question}
        </Question>
        Answer:
        """)
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    def ask(self, query: str):
        response = self.chain.invoke(query)
        return response
if __name__ == "__main__":
    rag = RAGService()
    print("Bot is ready! (Type 'exit' to stop)")  
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break       
        try:
            response = rag.ask(user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"Error: {e}")