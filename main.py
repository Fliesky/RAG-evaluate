import logging
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from ragflow_sdk import RAGFlow, Session

from data import gpt4o_gts, gpt4o_questions, gts, questions


def init_chat_model() -> ChatOllama:
    return ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_URL)


def init_embd_model() -> LangchainEmbeddingsWrapper:
    ollama = OllamaEmbeddings(model=OLLAMA_EMBD_MODEL, base_url=OLLAMA_URL)

    return LangchainEmbeddingsWrapper(ollama)


def init_ragflow_session() -> Session:
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ragflow = RAGFlow(api_key=RAGFLOW_APIKEY, base_url=RAGFLOW_URL)
    datasets = ragflow.list_datasets(name=RAGFLOW_DATASET)
    dataset_ids = []
    for dataset in datasets:
        dataset_ids.append(dataset.id)

    try:
        assistant = ragflow.list_chats(name=RAGFLOW_ASSISTANT)[0]
    except Exception:
        # llm = Chat.LLM(
        #     ragflow,
        #     {
        #         "model_name": None,
        #         "temperature": 0.1,
        #         "top_p": 0.3,
        #         "presence_penalty": 0,
        #         "frequency_penalty": 0,
        #         "max_tokens": 8192
        #     }
        # )
        # assistant = ragflow.create_chat(RAGFLOW_ASSISTANT, dataset_ids=dataset_ids, llm=llm)

        logging.ERROR("Assistant does not exist.")
        exit()

    session = assistant.create_session(f"{RAGFLOW_SESSION}-{current_time}")

    return session


def remove_tags(input: str) -> str:
    pattern1 = r"<think>.*?</think>"
    output = re.sub(pattern1, "", input, flags=re.DOTALL)
    pattern2 = r"##\d+?\$\$"
    output = re.sub(pattern2, "", output, flags=re.DOTALL)

    return output


def get_eval_dataset(session: Session) -> EvaluationDataset:
    samples = []
    for question, truth in QA_DATA[QA_SOURCE]:
        res = session.ask(question, stream=True)
        answer = ""
        contexts = []
        for ans in res:
            answer = ans.content
            if ans.reference is not None:
                for ref in ans.reference:
                    contexts.append(ref["content"])

    answer = remove_tags(answer)

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=answer,
        reference=truth
    )
    logging.warning(f"question: {sample.user_input}")
    logging.warning(f"context: {sample.retrieved_contexts}")
    logging.warning(f"answer: {sample.response}")
    logging.warning(f"truth: {sample.reference}")

    samples.append(sample)
    eval_dataset = EvaluationDataset(samples=samples)

    return eval_dataset


def eval_performance(
    chat: ChatOllama, embd: LangchainEmbeddingsWrapper, eval_dataset: EvaluationDataset
) -> None:
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_similarity
    ]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=chat,
        embeddings=embd
    )

    logging.warning(f"result: {result}")


def main() -> None:
    chat = init_chat_model()
    embd = init_embd_model()
    session = init_ragflow_session()
    eval_dataset = get_eval_dataset(session)
    eval_performance(chat, embd, eval_dataset)


if __name__ == "__main__":
    # ENV
    load_dotenv()
    OLLAMA_URL = os.getenv("OLLAMA_URL")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")
    OLLAMA_EMBD_MODEL = os.getenv("OLLAMA_EMBD_MODEL")

    RAGFLOW_URL = os.getenv("RAGFLOW_URL")
    RAGFLOW_APIKEY = os.getenv("RAGFLOW_APIKEY")
    RAGFLOW_DATASET = os.getenv("RAGFLOW_DATASET")
    RAGFLOW_ASSISTANT = os.getenv("RAGFLOW_ASSISTANT")
    RAGFLOW_SESSION = os.getenv("RAGFLOW_SESSION")

    # QA
    QA_DATA = {
        "default": (questions, gts),
        "gpt4o": (gpt4o_questions, gpt4o_gts)
    }
    # QA_SOURCE = "default"
    QA_SOURCE = "gpt4o"

    # LOG
    log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(Path(__file__).parent / "ragas.log", mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    main()
