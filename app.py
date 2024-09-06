import gradio as gr
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the NASA-specific bi-encoder model and tokenizer
bi_encoder_model_name = "nasa-impact/nasa-smd-ibm-st-v2"
bi_tokenizer = AutoTokenizer.from_pretrained(bi_encoder_model_name)
bi_model = AutoModel.from_pretrained(bi_encoder_model_name)

# Set up OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Define a system message to introduce Exos
system_message = "You are Exos, a helpful assistant specializing in Exoplanet research. Provide detailed and accurate responses related to Exoplanet research."

def encode_text(text):
    inputs = bi_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bi_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()  # Ensure the output is 2D

def retrieve_relevant_context(user_input, context_texts):
    user_embedding = encode_text(user_input).reshape(1, -1)
    context_embeddings = np.array([encode_text(text) for text in context_texts])
    context_embeddings = context_embeddings.reshape(len(context_embeddings), -1)  # Flatten each embedding
    similarities = cosine_similarity(user_embedding, context_embeddings).flatten()
    most_relevant_idx = np.argmax(similarities)
    return context_texts[most_relevant_idx]

def generate_response(user_input, relevant_context="", max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):
    if relevant_context:
        combined_input = f"Context: {relevant_context}\nQuestion: {user_input}\nAnswer:"
    else:
        combined_input = f"Question: {user_input}\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": combined_input}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response.choices[0].message.content.strip()

def chatbot(user_input, context="", use_encoder=False, max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):
    if use_encoder and context:
        context_texts = context.split("\n")
        relevant_context = retrieve_relevant_context(user_input, context_texts)
    else:
        relevant_context = ""
    response = generate_response(user_input, relevant_context, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your message here...", label="Your Question"),
        gr.Textbox(lines=5, placeholder="Enter context here, separated by new lines...", label="Context"),
        gr.Checkbox(label="Use NASA SMD Bi-Encoder for Context"),
        gr.Slider(50, 500, value=150, step=10, label="Max Tokens"),
        gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.1, label="Top-p"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Frequency Penalty"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Presence Penalty")
    ],
    outputs=gr.Textbox(label="Exos says..."),
    title="Exos - Your Exoplanet Research Assistant",
    description="Exos is a helpful assistant specializing in Exoplanet research. Provide context to get more refined and relevant responses.",
)

# Launch the interface
iface.launch(share=True)