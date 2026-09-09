from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    relevant_pages: tuple[int, ...]


EVALUATION_DATASET = [
    EvaluationQuestion(
        question="How does the LangChain Orchestrator process incoming requests?",
        relevant_pages=(20, 21),
    ),
    EvaluationQuestion(
        question="What information does the LangChain Orchestrator retrieve from DynamoDB?",
        relevant_pages=(21,),
    ),
    EvaluationQuestion(
        question="How does the solution retrieve documents for a knowledge-base-enabled deployment?",
        relevant_pages=(21, 82, 150, 151),
    ),
    EvaluationQuestion(
        question="What is the purpose of the Workflow Builder?",
        relevant_pages=(12, 43),
    ),
    EvaluationQuestion(
        question="What is the purpose of the Agent Builder?",
        relevant_pages=(12, 39),
    ),
    EvaluationQuestion(
        question="What information is included in the prompt for a RAG deployment?",
        relevant_pages=(82, 154, 184),
    ),
    EvaluationQuestion(
        question="How does query rephrasing help the AI model?",
        relevant_pages=(154,),
    ),
    EvaluationQuestion(
        question="How is conversation history maintained?",
        relevant_pages=(157, 169),
    ),
    EvaluationQuestion(
        question="What AWS services are involved in monitoring the solution?",
        relevant_pages=(32, 43, 118, 163),
    ),
    EvaluationQuestion(
        question="What AWS Regions are supported by the solution?",
        relevant_pages=(45,),
    ),
]
