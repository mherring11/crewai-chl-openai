import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Updated to use the ChatOpenAI wrapper for OpenAI integration

# Load the OpenAI API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class QuestionAnalysisAgents:
    def __init__(self):
        # Initialize ChatOpenAI with the GPT-4 model and API key
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4"  # Set the model to GPT-4
        )
        print("OpenAI GPT-4 model initialized.")

    def qc_testing_agent(self, input_text):
        """Simulates an agent to answer questions based on input text."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{input_text}\n\nRespond to the above questions:"}
        ]
        try:
            response = self.llm(messages)  # Use the correct call for ChatOpenAI
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "There was an error generating the response."

    def qc_auditor_agent(self, response_text):
        """Simulates an auditor agent to evaluate a response."""
        messages = [
            {"role": "system", "content": "You are an evaluator who scores responses based on relevance, completeness, accuracy, clarity, and conciseness."},
            {"role": "user", "content": f"Evaluate the following response and provide a score between 0 and 100: {response_text}"}
        ]
        try:
            audit_response = self.llm(messages)  # Use the correct call for ChatOpenAI
            return audit_response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating audit response: {e}")
            return "There was an error generating the audit response."
