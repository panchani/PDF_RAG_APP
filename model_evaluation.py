import os
import asyncio
import json
from dotenv import load_dotenv

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from ragas.embeddings import HuggingFaceEmbeddings

load_dotenv()


async def main():

    client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    llm = llm_factory(
        model="openai/gpt-oss-20b",
        client=client
    )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Load dataset from JSON
    # -----------------------------
    with open("rag_evaluation_dataset.json", "r") as f:
        test_cases = json.load(f)

    # -----------------------------
    # Loop through test cases
    # -----------------------------
    for i, case in enumerate(test_cases):

        print("\n==============================")
        print(f"Running Test Case {i+1}")
        print(f"Question: {case['user_input']}")
        print("==============================")

        user_input = case["user_input"]
        response = case["response"]
        retrieved_contexts = case["retrieved_contexts"]
        reference = case["reference"]

        # ---------------------
        # Faithfulness
        # ---------------------
        try:
            print("\nRunning Faithfulness...")
            faithfulness = Faithfulness(llm=llm)

            result = await faithfulness.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts
            )

            print("Faithfulness Score:", result.value)

        except Exception as e:
            print("Faithfulness FAILED:", e)

        # ---------------------
        # Answer Relevancy
        # ---------------------
        try:
            print("\nRunning AnswerRelevancy...")
            relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

            result = await relevancy.ascore(
                user_input=user_input,
                response=response
            )

            print("AnswerRelevancy Score:", result.value)

        except Exception as e:
            print("AnswerRelevancy FAILED:", e)

        # ---------------------
        # Context Precision
        # ---------------------
        try:
            print("\nRunning ContextPrecision...")
            precision = ContextPrecision(llm=llm)

            result = await precision.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts
            )

            print("ContextPrecision Score:", result.value)

        except Exception as e:
            print("ContextPrecision FAILED:", e)

        # ---------------------
        # Context Recall
        # ---------------------
        try:
            print("\nRunning ContextRecall...")
            recall = ContextRecall(llm=llm)

            result = await recall.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts
            )

            print("ContextRecall Score:", result.value)

        except Exception as e:
            print("ContextRecall FAILED:", e)


asyncio.run(main())