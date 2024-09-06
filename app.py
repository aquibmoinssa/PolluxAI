import gradio as gr
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import io
import tempfile
from astroquery.nasa_ads import ADS
import pyvo as vo

# Load the NASA-specific bi-encoder model and tokenizer
bi_encoder_model_name = "nasa-impact/nasa-smd-ibm-st-v2"
bi_tokenizer = AutoTokenizer.from_pretrained(bi_encoder_model_name)
bi_model = AutoModel.from_pretrained(bi_encoder_model_name)

# Set up OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Set up NASA ADS token
ADS.TOKEN = os.getenv('ADS_API_KEY')  # Ensure your ADS API key is stored in environment variables

# Define a system message to introduce Exos
system_message = "You are ExosAI, a helpful assistant specializing in Astrophysics and Exoplanet research. Provide detailed and accurate responses related to Astrophysics and Exoplanet research."

def encode_text(text):
    inputs = bi_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bi_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

def retrieve_relevant_context(user_input, context_texts):
    user_embedding = encode_text(user_input).reshape(1, -1)
    context_embeddings = np.array([encode_text(text) for text in context_texts])
    context_embeddings = context_embeddings.reshape(len(context_embeddings), -1)
    similarities = cosine_similarity(user_embedding, context_embeddings).flatten()
    most_relevant_idx = np.argmax(similarities)
    return context_texts[most_relevant_idx]

def fetch_nasa_ads_references(prompt):
    try:
        # Use the entire prompt for the query
        simplified_query = prompt

        # Query NASA ADS for relevant papers
        papers = ADS.query_simple(simplified_query)
        
        if not papers or len(papers) == 0:
            return [("No results found", "N/A", "N/A")]
        
        # Include authors in the references
        references = [
            (
                paper['title'][0], 
                ", ".join(paper['author'][:3]) + (" et al." if len(paper['author']) > 3 else ""), 
                paper['bibcode']
            ) 
            for paper in papers[:5]  # Limit to 5 references
        ]
        return references
    
    except Exception as e:
        return [("Error fetching references", str(e), "N/A")]

def fetch_exoplanet_data():
    # Connect to NASA Exoplanet Archive TAP Service
    tap_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")

    # Query to fetch all columns from the pscomppars table
    ex_query = """
        SELECT TOP 10 *
        FROM pscomppars
    """
    # Execute the query
    qresult = tap_service.search(ex_query)

    # Convert to a Pandas DataFrame
    ptable = qresult.to_table()
    exoplanet_data = ptable.to_pandas()

    return exoplanet_data

def generate_response(user_input, relevant_context="", references=[], max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):
    if relevant_context:
        combined_input = f"Context: {relevant_context}\nQuestion: {user_input}\nAnswer (please organize the answer in a structured format with topics and subtopics):"
    else:
        combined_input = f"Question: {user_input}\nAnswer (please organize the answer in a structured format with topics and subtopics):"
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
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
    
    # Append references to the response
    if references:
        response_content = response.choices[0].message.content.strip()
        references_text = "\n\nADS References:\n" + "\n".join(
            [f"- {title} by {authors} (Bibcode: {bibcode})" for title, authors, bibcode in references]
        )
        return f"{response_content}\n{references_text}"
    
    return response.choices[0].message.content.strip()

def export_to_word(response_content):
    doc = Document()
    doc.add_heading('AI Generated SCDD', 0)
    for line in response_content.split('\n'):
        doc.add_paragraph(line)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_file.name)
    
    return temp_file.name

def chatbot(user_input, context="", use_encoder=False, max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):
    if use_encoder and context:
        context_texts = context.split("\n")
        relevant_context = retrieve_relevant_context(user_input, context_texts)
    else:
        relevant_context = ""

    # Fetch NASA ADS references using the full prompt
    references = fetch_nasa_ads_references(user_input)

    # Generate response from GPT-4
    response = generate_response(user_input, relevant_context, references, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)

    # Export the response to a Word document
    word_doc_path = export_to_word(response)

    # Fetch exoplanet data
    exoplanet_data = fetch_exoplanet_data()
    
    # Embed Miro iframe
    iframe_html = """
    <iframe width="768" height="432" src="https://miro.com/app/live-embed/uXjVKuVTcF8=/?moveToViewport=-331,-462,5434,3063&embedId=710273023721" frameborder="0" scrolling="no" allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen></iframe>
    """
    
    mapify_button_html = """
    <style>
        .mapify-button {
            background: linear-gradient(135deg, #1E90FF 0%, #87CEFA 100%);
            border: none;
            color: white;
            padding: 15px 35px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 18px;
            font-weight: bold;
            margin: 20px 2px;
            cursor: pointer;
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .mapify-button:hover {
            background: linear-gradient(135deg, #4682B4 0%, #1E90FF 100%);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            transform: scale(1.05);
        }
    </style>
    <a href="https://mapify.so/app/new" target="_blank">
        <button class="mapify-button">Create Mind Map on Mapify</button>
    </a>
    """
    return response, iframe_html, mapify_button_html, word_doc_path, exoplanet_data

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your Science Goal here...", label="Prompt ExosAI"),
        gr.Textbox(lines=5, placeholder="Enter some context here...", label="Context"),
        gr.Checkbox(label="Use NASA SMD Bi-Encoder for Context"),
        gr.Slider(50, 1000, value=150, step=10, label="Max Tokens"),
        gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.1, label="Top-p"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Frequency Penalty"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Presence Penalty")
    ],
    outputs=[
        gr.Textbox(label="ExosAI finds..."),
        gr.HTML(label="Miro"),
        gr.HTML(label="Generate Mind Map on Mapify"),
        gr.File(label="Download SCDD", type="filepath"),
        gr.Dataframe(label="Exoplanet Data Table")
    ],
    title="ExosAI - NASA SMD SCDD AI Assistant [version-0.4a]",
    description="ExosAI is an AI-powered assistant for generating and visualising HWO Science Cases",
)

iface.launch(share=True)