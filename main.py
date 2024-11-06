import os
import re
import logging
from fpdf import FPDF
import matplotlib.pyplot as plt
from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.pdf_reader import PDFReader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logging.error("OPENAI_API_KEY not found in environment.")
else:
    logging.info("Environment variables loaded successfully.")

class QuestionAnalysisAgents:
    def __init__(self):
        logging.info("Initializing Analysis Agents...")
        # Specify the "gpt-4o-mini" model
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        logging.info("GPT-4o mini model initialized.")

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
            match = re.search(r'Score:\s*(\d+)', content)
            if match:
                return int(match.group(1))
            else:
                logging.warning(f"Score not found in {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error reading score from {filepath}: {e}")
    return None

def analyze_questions(pdf_files):
    agents = QuestionAnalysisAgents()
    qc_agent = agents.qc_testing_agent()
    auditor_agent = agents.qc_auditor_agent()
    
    summary = {}

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            logging.info(f"File exists: {pdf_path}")
            try:
                text = PDFReader.read_pdf(pdf_path)

                output_file = f"{os.path.splitext(pdf_path)[0]}_analysis.txt"
                score_file = f"{os.path.splitext(pdf_path)[0]}_score.txt"
                with open(output_file, 'w') as f, open(score_file, 'w') as sf:
                    logging.info(f"Processing {pdf_path}")

                    qc_result = qc_agent.execute_task(text)
                    f.write(f"--- QC Testing Agent ({pdf_path}) ---\n")
                    f.write(f"Questions and Answers:\n{text}\n")
                    f.write(f"Simulated Response:\n{qc_result}\n")

                    audit_input = f"Evaluate the following response and provide a score between 0 and 100: {qc_result}"
                    audit_result = auditor_agent.execute_task(audit_input)

                    f.write("\n--- Auditor Agent ---\n")
                    f.write(f"Evaluation and Score:\n{audit_result}\n")

                    score_match = re.search(r'\b(\d+)\b', audit_result)
                    if score_match:
                        score = int(score_match.group(1))
                        summary[pdf_path] = score
                        sf.write(f"Score: {score}\n")
                        sf.write(f"Evaluation Summary:\n{audit_result}\n")
                        logging.info(f"Score for {pdf_path} written to {score_file}")
                    else:
                        summary[pdf_path] = "Score not found"
                        sf.write("Score: Not found\n")
                        sf.write(f"Evaluation Summary:\n{audit_result}\n")
                        logging.warning(f"No score found in audit result for {pdf_path}")

            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                summary[pdf_path] = "Error in processing"
        else:
            logging.warning(f"File not found: {pdf_path}")
            summary[pdf_path] = "File not found"

    logging.info("\n--- Summary of Scores ---")
    for pdf, score in summary.items():
        logging.info(f"{pdf}: {score}")

    generate_graph(summary)
    process_scores_and_generate_reports('./pdfs')  # Automatically run the PDF generation

def read_scores_from_files(directory):
    """Reads scores from all *_score.txt files in the given directory and returns a dictionary."""
    score_summary = {}

    for filename in os.listdir(directory):
        if filename.endswith("_score.txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as file:
                    content = file.read()
                    match = re.search(r'Score:\s*(\d+)', content)
                    if match:
                        score = int(match.group(1))
                        score_summary[filename] = {
                            "score": score,
                            "content": content
                        }
                    else:
                        logging.warning(f"Score not found in {filename}")
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    
    return score_summary

def export_low_score_to_pdf(filename, score_data):
    """Creates a PDF with details for entries where the score is below 96."""
    output_dir = "./low_score_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)

    pdf.cell(0, 10, f"Low Score Report for {filename}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Score: {score_data['score']}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Evaluation Summary:", ln=True)
    pdf.multi_cell(0, 10, score_data["content"])

    # Save PDF in the low_score_reports directory
    output_pdf_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_low_score_report.pdf")
    pdf.output(output_pdf_path)
    logging.info(f"Low score report generated: {output_pdf_path}")

def process_scores_and_generate_reports(directory):
    """Processes scores from score files and generates reports for scores below 96."""
    scores = read_scores_from_files(directory)
    
    for filename, score_data in scores.items():
        if score_data["score"] < 96:
            export_low_score_to_pdf(filename, score_data)
        else:
            logging.info(f"Score for {filename} is {score_data['score']} - no PDF generated.")

def generate_graph(summary):
    pdf_files = [pdf for pdf, score in summary.items() if isinstance(score, int)]
    scores = [score for score in summary.values() if isinstance(score, int)]
    
    if not scores:
        logging.warning("No valid scores found to plot.")
        return

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
