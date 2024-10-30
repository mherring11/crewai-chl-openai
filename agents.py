import os
import re
import matplotlib.pyplot as plt
from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.pdf_reader import PDFReader

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("Environment variables loaded.")
print(f"OPENAI_API_KEY: {openai_api_key}")

class QuestionAnalysisAgents:
    def __init__(self):
        print("Initializing Analysis Agents...")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")  # Using GPT-4
        print("GPT-4 model initialized.")

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
            goal="Evaluate the accuracy of answers based on the provided content.",
            backstory="Reads the simulated answer and rates it based on relevance, accuracy, and clarity.",
            llm=self.llm,
            verbose=True
        )

def extract_score_from_file(filepath):
    """Extracts the score from the auditor agent section of the analysis file."""
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            # Use regex to search for "Score: X" where X is a number
            match = re.search(r'Score:\s*(\d+)', content)
            if match:
                return int(match.group(1))
            else:
                print(f"Score not found in {filepath}")
    except Exception as e:
        print(f"Error reading score from {filepath}: {e}")
    return None

def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()
    
    # Initialize a summary dictionary to store scores for each PDF
    summary = {}

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"File exists: {pdf_path}")
            try:
                # Read the entire text from PDF
                text = PDFReader.read_pdf(pdf_path)
                print(f"Extracted text from {pdf_path}:\n{text}...")

                # Prepare output file for storing questions, answers, and scores
                output_file = f"{os.path.splitext(pdf_path)[0]}_analysis.txt"
                with open(output_file, 'w') as f:
                    print(f"Processing {pdf_path}")

                    # Use the QC Testing Agent to get a simulated response
                    qc_result = qc_agent.execute_task(text)
                    f.write(f"--- QC Testing Agent ({pdf_path}) ---\n")
                    f.write(f"Questions and Answers:\n{text}\n")
                    f.write(f"Simulated Response:\n{qc_result}\n")

                    # Use the QC Auditor Agent to evaluate the response
                    audit_input = f"Evaluate the following response and provide a score between 0 and 100: {qc_result}"
                    audit_result = auditor_agent.execute_task(audit_input)
                    f.write("\n--- Auditor Agent ---\n")
                    f.write(f"Evaluation and Score:\n{audit_result}\n")

                    print(f"Results written to {output_file}")

                # Extract the score from the output file
                score = extract_score_from_file(output_file)
                if score is not None:
                    summary[pdf_path] = score
                else:
                    summary[pdf_path] = "Score not found"

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                summary[pdf_path] = "Error in processing"
        else:
            print(f"File not found: {pdf_path}")
            summary[pdf_path] = "File not found"

    # Display the summary of scores
    print("\n--- Summary of Scores ---")
    for pdf, score in summary.items():
        print(f"{pdf}: {score}")

    # Create a bar chart if scores are available
    generate_graph(summary)

def generate_graph(summary):
    # Filter out entries where the score was not found
    pdf_files = [pdf for pdf, score in summary.items() if isinstance(score, int)]
    scores = [score for score in summary.values() if isinstance(score, int)]
    
    if not scores:
        print("No valid scores found to plot.")
        return

    # Plotting the scores as a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(pdf_files, scores, color='skyblue')
    plt.xlabel('Score')
    plt.title('Summary of Scores for Each PDF')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pdf_files = [
        "pdfs/Chl_chatbot_test_questions_accessibility.pdf",
        "pdfs/Chl_chatbot_test_questions_app_development.pdf",
        "pdfs/Chl_chatbot_test_questions_maintenance_support.pdf",
        "pdfs/Chl_chatbot_test_questions_marketing_site.pdf",
        "pdfs/Chl_chatbot_test_questions_motion_graphics.pdf",
        "pdfs/Chl_chatbot_test_questions_seo.pdf",
    ]

    analyze_questions(pdf_files)
