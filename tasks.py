from crewai import Task
from textwrap import dedent

class QuestionAnalysisTasks:
    def simulate_answer(self, agent, question):
        return Task(
            description=dedent(f"Simulate an answer for the question: '{question}'"),
            agent=agent
        )

    def evaluate_answer(self, agent, provided_answer, simulated_answer):
        return Task(
            description=dedent(f"Evaluate the correctness of the provided answer '{provided_answer}' compared to '{simulated_answer}'."),
            agent=agent
        )