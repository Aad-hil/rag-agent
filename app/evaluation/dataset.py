from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    relevant_pages: tuple[int, ...]
    reference_answer: str = ""


EVALUATION_DATASET = [
    EvaluationQuestion(
        question="How does the LangChain Orchestrator process incoming requests?",
        relevant_pages=(20, 21),
        reference_answer=(
            "Incoming requests go from API Gateway to an SQS queue and then "
            "to the LangChain Orchestrator. The orchestrator retrieves LLM "
            "configuration and session information, optionally retrieves "
            "document excerpts from the knowledge base, combines the query, "
            "chat history, and retrieved context into a final prompt, sends "
            "it to the LLM, and streams the response back through the API "
            "Gateway WebSocket."
        ),
    ),
    EvaluationQuestion(
        question="What information does the LangChain Orchestrator retrieve from DynamoDB?",
        relevant_pages=(21,),
        reference_answer=(
            "The LangChain Orchestrator retrieves the configured LLM options "
            "and necessary session information, including chat history."
        ),
    ),
    EvaluationQuestion(
        question="How does the solution retrieve documents for a knowledge-base-enabled deployment?",
        relevant_pages=(21, 82, 150, 151),
        reference_answer=(
            "For a RAG deployment, the LangChain Orchestrator uses Amazon "
            "Kendra or Amazon Bedrock Knowledge Bases to run a search query "
            "and retrieve document excerpts."
        ),
    ),
    EvaluationQuestion(
        question="What is the purpose of the Workflow Builder?",
        relevant_pages=(12, 43),
        reference_answer=(
            "The Workflow Builder creates supervisor agents that orchestrate "
            "multiple Agent Builder agents using the Agents as Tools "
            "delegation pattern, enabling complex multi-agent workflows and "
            "reuse of existing Agent Builder deployments."
        ),
    ),
    EvaluationQuestion(
        question="What is the purpose of the Agent Builder?",
        relevant_pages=(12, 39),
        reference_answer=(
            "Agent Builder provides a platform for creating, deploying, and "
            "managing production-ready AI agents on Amazon Bedrock AgentCore, "
            "with configuration, MCP integration, and memory-management "
            "capabilities."
        ),
    ),
    EvaluationQuestion(
        question="What information is included in the prompt for a RAG deployment?",
        relevant_pages=(82, 154, 184),
        reference_answer=(
            "The prompt can contain the user's input, conversation history, "
            "and retrieved knowledge-base context. For RAG deployments, "
            "the {context} placeholder is required and is replaced with "
            "document excerpts retrieved from the knowledge base."
        ),
    ),
    EvaluationQuestion(
        question="How does query rephrasing help the AI model?",
        relevant_pages=(154,),
        reference_answer=(
            "Rephrasing or disambiguating the user's query can help the model "
            "better understand the user's intent and potentially produce "
            "more accurate responses."
        ),
    ),
    EvaluationQuestion(
        question="How is conversation history maintained in the LangChain implementation?",
        relevant_pages=(157, 169),
        reference_answer=(
            "Conversation history is maintained through the solution's "
            "conversation-memory implementation. The configuration can use "
            "DynamoDB-based memory, while the LangChain implementation uses "
            "RunnableWithMessageHistory to maintain conversation history "
            "with LCEL chains."
        ),
    ),
    EvaluationQuestion(
        question="What AWS services are involved in monitoring the solution?",
        relevant_pages=(32, 43, 118, 163),
        reference_answer=(
            "Amazon CloudWatch provides operational metrics, dashboards, and "
            "log analysis; AWS Systems Manager Application Manager provides "
            "an application-level view for monitoring resources, costs, and "
            "logs. Agent Builder also uses ADOT for tracing and structured "
            "logging integrated with CloudWatch Transaction Search."
        ),
    ),
    EvaluationQuestion(
        question="What AWS Regions are supported by the solution?",
        relevant_pages=(45,),
        reference_answer=(
            "The supported Regions listed in the document include US East "
            "(Ohio), US East (N. Virginia), US West (Northern California), "
            "US West (Oregon), Canada (Central), Europe (Frankfurt), "
            "Europe (Ireland), Europe (London), Europe (Milan), Europe "
            "(Paris), Europe (Stockholm), Asia Pacific (Mumbai), Asia "
            "Pacific (Seoul), Asia Pacific (Singapore), Asia Pacific "
            "(Sydney), Asia Pacific (Tokyo), Middle East (Bahrain), and "
            "South America (São Paulo)."
        ),
    ),
]