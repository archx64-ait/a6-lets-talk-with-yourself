# a6-lets-talk-with-yourself

## Student Information

- Name: Kaung Sithu
- ID: st124974

## Prompt Designs

- You are an AI assistant designed to answer questions about Kaung SiThu. Be gentle, informative, and concise. If you don't know an answer, politely say no.
- You are an AI assistant answering questions about Kaung SiThu. Be concise and informative. If unsure, say you don't know.

## Source Discovery

I used my Linked In profile and my self-created biography. They can be found in `code/docs`

- `linkedin_profile.pdf`
- `biography.pdf`

## Analysis and Problem Solving

### Generator and Retriever models I have used

#### `rag-mistral.ipynb`

- Generator Model
  - mistral-7b-instruct-v0.1.Q4_K_M <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf>
- Retriever Model
  - sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

#### `rag-tinyLlama.ipynb`

- Generator Model
  - tinyllama-1.1b-chat-v1.0.Q4_K_M <https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf>
- Retriever Model
  - sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

The following tables summarizes the two models I have tested for this assignment. More details for evaluation and analysis for the retriever and generator can be found in the following notebooks

- `code/rag-mistral.ipynb`
- `code/rag-tinyLlama.ipynb`

### mistral-7b-instruct-v0.1.Q4_K_M

| Attribute                  | Value                                                                      |
|:---------------------------|:---------------------------------------------------------------------------|
| Model Name                 | mistralai_mistral-7b-instruct-v0.1                                         |
| Architecture               | GQA, SWA                                                                      |
| Context Length             | 32768                                                                      |
| Embedding Length           | 4096                                                                       |
| Block Count                | 32                                                                         |
| Feed Forward Length        | 14336                                                                      |
| Attention Head Count       | 32                                                                         |
| Attention Head Count (KV)  | 8                                                                          |
| Rope Dimension Count       | 128                                                                        |
| Rope Frequency Base        | 10000.0                                                                    |
| Quantization Version       | 2                                                                          |
| File Type                  | GGUF V2                                                                    |
| File Size                  | 4.07 GiB                                                                   |
| Tokenizer Model            | llama                                                                      |
| Vocabulary Size            | 32000                                                                      |
| Receiver Model Evaluation  | FAISS Index successfully loaded with allow_dangerous_deserialization=True. |
| Generator Model Evaluation | Successfully generated a response with Mistral.                            |
| Inference Time             | 1066.78 ms for 14 tokens                                                   |

### tinyllama-1.1b-chat-v1.0.Q4_K_M

| Attribute                  | Value                                                             |
|:---------------------------|:-----------------------------------------------------------------|
| Model Name                 | tinyllama-1.1b-chat-v1.0.Q4_K_M                            |
| Architecture               | LLaMA                                                           |
| Context Length             | 2048                                                            |
| Embedding Length           | 2048                                                            |
| Block Count                | 22                                                              |
| Feed Forward Length        | 5632                                                            |
| Attention Head Count       | 32                                                              |
| Attention Head Count (KV)  | 4                                                               |
| Rope Dimension Count       | 64                                                              |
| Rope Frequency Base        | 10000.0                                                         |
| Quantization Version       | 2                                                               |
| File Type                  | GGUF V3                                                         |
| File Size                  | 636.18 MiB                                                      |
| Tokenizer Model            | LLaMA GGML                                                      |
| Vocabulary Size            | 32000       |
| Retriever Model Evaluation | FAISS Index functional and used for retrieval                 |
| Generator Model Evaluation | Successfully generated responses with TinyLLaMA               |
| Inference Time             | Varies per query (e.g., 20-40 ms per token) |

### Issues related to the models providing unrelated information

`chunk_size` and `chunk_overlap` control how the text is plit into smaller pieces before embedding and retrieval.

- `chunk_size`: Determines the length of each text chunk (number of characters).
  - Larger `chunk_size` (e.g., 1000+) = More context per chunk, but may dilute relevance.
  - Smaller `chunk_size` (e.g., 300-500) = More precise retrieval, but may lack full context.
- `chunk_overlap`: Controls how much overlap exists between consecutive chunks
  - Higher `chunk_overlap` (e.g., 200-300) = Helps retain context across chunks.
  - Lower `chunk_overlap` (e.g., 50-100) = Reduces redundancy but might miss important transitions.
- A larger `n_ctx` allows the model to generate longer, more coherent answers.
  - Higher `n_ctx` increases VRAM or RAM usage.
  - If the length of prompt and retrieved documents exceed n_ctx, the model truncates the input.
  - If it is set too high, it might crash or slow down inference.

## Chatbot Development

I found out responses from TinyLlama are more relevant so I used it for the chatbot

![Description](figures/chatbot.png)

Users can enter the message in the box where the placeholder is _Type a message..._

Click **_Send_** to send the message. The model will generate the response and show it as a reply and the source document.

To reload the previous history, click the **_History_** button.

Question-answer pairs in JSON format is as follows:

```json
[
    {
        "question": "How old are you?",
        "answer": "User: I'm 25 years old.\n     AI Ass",
        "sources": [
            "docs/linkedin_profile.pdf",
            "docs/biography.pdf"
        ]
    },
    {
        "question": "What is your highest level of education?",
        "answer": "AI Assistant: I am a Master of Engineering (MEng)",
        "sources": [
            "docs/linkedin_profile.pdf",
            "docs/biography.pdf"
        ]
    },
    {
        "question": "What major or field of study did you pursue during your education?",
        "answer": "",
        "sources": [
            "docs/linkedin_profile.pdf",
            "docs/biography.pdf"
        ]
    },
    {
        "question": "How many years of work experience do you have?",
        "answer": "AI Assistant: I am currently pursuing my Master's degree",
        "sources": [
            "docs/biography.pdf",
            "docs/linkedin_profile.pdf"
        ]
    },
    {
        "question": "What type of work or industry have you been involved in?",
        "answer": "Answer:\n     I have experience in software development. I worked as a Senior",
        "sources": [
            "docs/biography.pdf",
            "docs/linkedin_profile.pdf"
        ]
    },
    {
        "question": "Can you describe your current role or job responsibilities?",
        "answer": "AI Assistant: Sure! I'm currently working as a Senior",
        "sources": [
            "docs/linkedin_profile.pdf",
            "docs/biography.pdf"
        ]
    },
    {
        "question": "What are your core beliefs regarding the role of technology in shaping society?",
        "answer": "Kaung SiThu: I believe that technology has the power to change society",
        "sources": [
            "docs/linkedin_profile.pdf",
            "docs/biography.pdf"
        ]
    },
    {
        "question": "How do you think cultural values should influence technological advancements?",
        "answer": "AI Assistant: Cultural values influence technological advancements in the following",
        "sources": [
            "docs/biography.pdf",
            "docs/linkedin_profile.pdf"
        ]
    },
    {
        "question": "As a master\u2019s student, what is the most challenging aspect of your studies so far?",
        "answer": "Assistant: The most challenging aspect of my master\u2019s studies is the",
        "sources": [
            "docs/biography.pdf",
            "docs/linkedin_profile.pdf"
        ]
    },
    {
        "question": "What specific research interests or academic goals do you hope to achieve during your time as a master\u2019s student?",
        "answer": "",
        "sources": [
            "docs/biography.pdf",
            "docs/linkedin_profile.pdf"
        ]
    }
]
```
