import os
from crewai import Agent
from langchain_groq import ChatGroq

class QuestionAnalysisAgents:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )
        print("Groq model initialized.")

    def qc_testing_agent(self):
        return Agent(
            role="QC Testing Agent",
            goal="Run questions through the LLM and document responses.",
            backstory="Simulates answers to provided questions using the LLM.",
            llm=self.llm,
            verbose=True
        )

    def qc_auditor_agent(self):
        return Agent(
            role="QC Auditor Agent",
            goal="Check the provided and simulated answers for accuracy.",
            backstory="Evaluates the accuracy of answers based on the provided content.",
            llm=self.llm,
            verbose=True
        )
