import os
import asyncio
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
    # Test Dataset (2 test cases)
    # -----------------------------
    test_cases = [

        {
            "user_input": "What architecture is proposed in this paper?",
            "response": "The proposed architecture is the Transformer model. It is described as following an overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.",
            "retrieved_contexts": [
                'Figure 1: The Transformer - model architecture.\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully\nconnected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,\nrespectively.\n3.1 Encoder and Decoder Stacks\nEncoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two\nsub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-\nwise fully connected feed-forward network. We employ a residual connection [11] around each of\nthe two sub-layers, followed by layer normalization [ 1]. That is, the output of each sub-layer is\nLayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer\nitself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding\nlayers, produce outputs of dimension dmodel = 512.\nDecoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two\nsub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head\nattention over the output of the encoder stack. Similar to the encoder, we employ residual connections\naround each of the sub-layers, followed by layer normalization. We also modify the self-attention\nsub-layer in the decoder stack to prevent positions from attending to subsequent positions. This\nmasking, combined with fact that the output embeddings are offset by one position, ensures that the\npredictions for position i can depend only on the known outputs at positions less than i.\n3.2 Attention\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,\nwhere the query, keys, values, and output are all vectors. The output is computed as a weighted sum\n3', 'output values. These are concatenated and once again projected, resulting in the final values, as\ndepicted in Figure 2.\nMulti-head attention allows the model to jointly attend to information from different representation\nsubspaces at different positions. With a single attention head, averaging inhibits this.\nMultiHead(Q, K, V ) = Concat(head 1, ..., headh)W O\nwhere headi = Attention(QW Q\ni , KW K\ni , V W V\ni )\nWhere the projections are parameter matricesW Q\ni ∈ Rdmodel×dk, W K\ni ∈ Rdmodel×dk, W V\ni ∈ Rdmodel×dv\nand W O ∈ Rhdv×dmodel.\nIn this work we employ h = 8 parallel attention layers, or heads. For each of these we use\ndk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost\nis similar to that of single-head attention with full dimensionality.\n3.2.3 Applications of Attention in our Model\nThe Transformer uses multi-head attention in three different ways:\n• In "encoder-decoder attention" layers, the queries come from the previous decoder layer,\nand the memory keys and values come from the output of the encoder. This allows every\nposition in the decoder to attend over all positions in the input sequence. This mimics the\ntypical encoder-decoder attention mechanisms in sequence-to-sequence models such as\n[38, 2, 9].\n• The encoder contains self-attention layers. In a self-attention layer all of the keys, values\nand queries come from the same place, in this case, the output of the previous layer in the\nencoder. Each position in the encoder can attend to all positions in the previous layer of the\nencoder.\n• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to\nall positions in the decoder up to and including that position. We need to prevent leftward\ninformation flow in the decoder to preserve the auto-regressive property. We implement this\ninside of scaled dot-product attention by masking out (setting to −∞) all values in the input\nof the softmax which correspond to illegal connections. See Figure 2.\n3.3 Position-wise Feed-Forward Networks\nIn addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully\nconnected feed-forward network, which is applied to each position separately and identically. This\nconsists of two linear transformations with a ReLU activation in between.\nFFN(x) = max(0 , xW1 + b1)W2 + b2 (2)\nWhile the linear transformations are the same across different positions, they use different parameters\nfrom layer to layer. Another way of describing this is as two convolutions with kernel size 1.\nThe dimensionality of input and output is dmodel = 512 , and the inner-layer has dimensionality\ndf f = 2048.\n3.4 Embeddings and Softmax\nSimilarly to other sequence transduction models, we use learned embeddings to convert the input\ntokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor-\nmation and softmax function to convert the decoder output to predicted next-token probabilities. In\nour model, we share the same weight matrix between the two embedding layers and the pre-softmax\nlinear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel.\n5', 'of continuous representations z = ( z1, ..., zn). Given z, the decoder then generates an output\nsequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive\n[10], consuming the previously generated symbols as additional input when generating the next.\n2'
            ],
            "reference": "The Transformer architecture is proposed."
        },

        {
            "user_input": "Who wrote the book Pride and Prejudice?",
            "response": "Pride and Prejudice was written by Jane Austen.",
            "retrieved_contexts": [
                "Pride and Prejudice is a novel written by Jane Austen and first published in 1813."
            ],
            "reference": "Jane Austen wrote Pride and Prejudice."
        }

    ]

    # -----------------------------
    # Loop through test cases
    # -----------------------------
    for i, case in enumerate(test_cases):

        print(f"\n==============================")
        print(f"Running Test Case {i+1}")
        print(f"Question: {case['user_input']}")
        print(f"==============================")

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