import os
from crewai import Agent, Crew
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.pdf_reader import PDFReader

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

print("Environment variables loaded.")
print(f"GROQ_API_KEY: {groq_api_key}")

class QuestionAnalysisAgents:
    def __init__(self):
        print("Initializing Analysis Agents...")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-70b-8192"
        )
        print("Groq model initialized with provided API key.")

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
            goal="Evaluate the provided and simulated answers for accuracy and assign a score between 0 and 100.",
            backstory="Reads the simulated answer and rates it based on relevance, completeness, accuracy, clarity, and conciseness.",
            llm=self.llm,
            verbose=True
        )

def chunk_text(text, chunk_size=3000):
    """Splits the text into smaller chunks to respect token limits."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"File exists: {pdf_path}")
            try:
                text = PDFReader.read_pdf(pdf_path)
                print(f"Extracted text from {pdf_path} (first 500 characters):\n{text[:500]}...")

                text_chunks = chunk_text(text)

                for idx, chunk in enumerate(text_chunks):
                    print(f"Processing chunk {idx + 1}/{len(text_chunks)} for {pdf_path}")

                    input_text = f"{chunk}"

                    qc_result = qc_agent.execute_task(input_text)
                    print(f"\n--- QC Testing Agent (Chunk {idx + 1}/{len(text_chunks)} of {pdf_path}) ---")
                    print(f"Questions and Answers:\n{input_text}\n")
                    print(f"Simulated Response:\n{qc_result}")

                    audit_input = f"Evaluate the following response and provide a score between 0 and 100 based on relevance, completeness, accuracy, clarity, and conciseness: {qc_result}"
                    audit_result = auditor_agent.execute_task(audit_input)
                    print(f"\n--- Auditor Agent (Chunk {idx + 1}/{len(text_chunks)} of {pdf_path}) ---")
                    print(f"Evaluation and Score:\n{audit_result}")

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        else:
            print(f"File not found: {pdf_path}")

if __name__ == "__main__":
    pdf_files = [
        "pdfs/Chl_chatbot_test_questions_accessibility.pdf",
        "pdfs/Chl_chatbot_test_questions_app_development.pdf",
        "pdfs/Chl_chatbot_test_questions_maintenance_support.pdf",
        "pdfs/Chl_chatbot_test_questions_marketing_site.pdf",
        "pdfs/Chl_chatbot_test_questions_motion_graphics.pdf",
        "pdfs/Chl_chatbot_test_questions_seo.pdf",
    ]

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"File exists: {pdf_path}")
        else:
            print(f"File not found: {pdf_path}")

    analyze_questions(pdf_files)
