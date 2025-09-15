
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import pandas as pd
from pinecone import Pinecone
#import logging
import re

from utils.ads_references import extract_keywords_with_gpt, fetch_nasa_ads_references 
from utils.data_insights import fetch_exoplanet_data, generate_data_insights
from utils.gen_doc import export_to_word
from utils.extract_table import extract_table_from_response, gpt_response_to_dataframe


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, ContextRelevance, Faithfulness, ResponseRelevancy, FactualCorrectness

# Load the NASA-specific bi-encoder model and tokenizer
bi_encoder_model_name = "nasa-impact/nasa-smd-ibm-st-v2"
bi_tokenizer = AutoTokenizer.from_pretrained(bi_encoder_model_name)
bi_model = AutoModel.from_pretrained(bi_encoder_model_name)

# Set up OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Pinecone setup
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index_name = "scdd-index"
index = pc.Index(index_name)

# Define system message with instructions
system_message = """
You are ExosAI, an advanced assistant specializing in Exoplanet and Astrophysics research.

Generate a **detailed and structured** response based on the given **retrieved context and user input**, incorporating key **observables, physical parameters, and technical requirements**. Organize the response into the following sections:

1. **Science Objectives**: Define key scientific objectives related to the science context and user input.
2. **Physical Parameters**: Outline the relevant physical parameters (e.g., mass, temperature, composition).
3. **Observables**: Specify the key observables required to study the science context.
4. **Description of Desired Observations**: Detail the observational techniques, instruments, or approaches necessary to gather relevant data.
5. **Observations Requirements Table**: Generate a table relevant to the Science Objectives, Physical Parameters, Observables and Description of Desired Observations with the following columns and at least 7 rows:
    - Wavelength Band: Should only be UV, Visible and Infrared).
    - Instrument: Should only be Imager, Spectrograph, Polarimeter and Coronagraph).
    - Necessary Values: The necessary values or parameters (wavelength range, spectral resolution where applicable, spatial resolution where applicable, contrast ratio where applicable).
    - Desired Values: The desired values or parameters (wavelength range, spectral resolution where applicable, spatial resolution where applicable).
    - Number of Objects Observed: Estimate the number of objects that need to be observed for a statistically meaningful result or for fulfilling the science objective.
    - Justification: Detailed scientific explanation of why these observations are important for the science objectives.
    - Comments: Additional notes or remarks regarding each observation.

#### **Table Format** 

| Wavelength Band      | Instrument                         | Necessary Values                   | Desired Values                  | Number of Objects Observed      | Justification     | Comments |
|----------------------|------------------------------------|------------------------------------|---------------------------------|---------------------------------|-------------------|----------|

#### **Guiding Constraints (Exclusions & Prioritization)**
- **Wavelength Band Restriction:** Only include **UV, Visible, and Infrared** bands.
- **Instrument Restriction:** Only include **Imager, Spectrograph, Polarimeter, and Coronagraph**.
- **Wavelength Limits:** Prioritize wavelengths between **100 nanometers (nm) and 3 micrometers (μm)**.
- **Allowed Instruments:** **Only include** observations from **direct imaging, spectroscopy, and polarimetry.** **Exclude** transit and radial velocity methods.
- **Exclusion of Existing Facilities:** **Do not reference** existing observatories such as JWST, Hubble, or ground-based telescopes. This work pertains to a **new mission**.
- **Spectral Resolution Constraint:** come up with an appropriate spectral resolution (**R**) depending on the requirements **.
- **Contrast Ratio:** come up with an appropriate contrast ratio depending on the requirements **.
- **Estimate the "Number of Objects Observed" based on the observational strategy, parameters, instruments, statistical requirements, and feasibility.**
- **Ensure that all parameters remain scientifically consistent.**
- **Include inline references in the Justification column wherever available **.
- **Pay close attention to the retrieved context**.

**Use this table format as a guideline, generate a detailed table dynamically based on the input.**. Ensure that all values align with the provided constraints and instructions.

**Include inline references wherever available**. Especially in the Justification column.

Ensure the response is **structured, clear, and observation requirements table follows this format**. **All included parameters must be scientifically consistent with each other.**
"""

# Function to encode query text
def encode_query(text):
    inputs = bi_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bi_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    embedding /= np.linalg.norm(embedding)
    return embedding.tolist()


# Context retrieval function using Pinecone
def retrieve_relevant_context(user_input, context_text, science_objectives="", top_k=5):
    query_text = f"Science Goal: {user_input}\nContext: {context_text}\nScience Objectives: {science_objectives}" if science_objectives else f"Science Goal: {user_input}\nContext: {context_text}"
    query_embedding = encode_query(query_text)

    # Pinecone query
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_context = "\n\n".join([match['metadata']['text'] for match in query_response.matches])

    if not retrieved_context.strip():
        return "No relevant context found for the query."

    return retrieved_context

def clean_retrieved_context(raw_context):
    # Remove unnecessary line breaks within paragraphs
    cleaned = raw_context.replace("-\n", "").replace("\n", " ")

    # Remove extra spaces clearly
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Return explicitly cleaned context
    return cleaned.strip()

def generate_response(user_input, science_objectives="", relevant_context="", references=[], max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):
    # Case 1: Both relevant context and science objectives are provided
    if relevant_context and science_objectives.strip():
        combined_input = f"Scientific Context: {relevant_context}\nUser Input: {user_input}\nScience Objectives (User Provided): {science_objectives}\n\nPlease generate only the remaining sections as per the defined format."
    
    # Case 2: Only relevant context is provided
    elif relevant_context:
        combined_input = f"Scientific Context: {relevant_context}\nUser Input: {user_input}\n\nPlease generate a full structured response, including Science Objectives."
    
    # Case 3: Neither context nor science objectives are provided
    elif science_objectives.strip():
        combined_input = f"User Input: {user_input}\nScience Objectives (User Provided): {science_objectives}\n\nPlease generate only the remaining sections as per the defined format."
    
    # Default: No relevant context or science objectives → Generate everything
    else:
        combined_input = f"User Input: {user_input}\n\nPlease generate a full structured response, including Science Objectives."
    
    response = client.chat.completions.create(
        model="gpt-4o",
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

    response_only = response.choices[0].message.content.strip()

    # ADS References appended separately
    references_text = ""
    if references:
        references_text = "\n\nADS References:\n" + "\n".join(
            [f"- {title} {authors} (Bibcode: {bibcode}) {pub} {pubdate}" 
             for title, abstract, authors, bibcode, pub, pubdate in references])

    # Full response (for Gradio display)
    full_response = response_only + references_text

    # Return two clearly separated responses
    return full_response, response_only
    
def chatbot(user_input, science_objectives="", context="", subdomain="", max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.0):

    
    yield "🔄 Connecting with Pinecone...", None, None, None, None, None, None
    
    pc_index_name = "scdd-index"
    yield f"Using Pinecone index: **{pc_index_name}**✅ ", None, None, None, None, None, None

    yield "🔎 Retrieving relevant context from Pinecone...", None, None, None, None, None, None
    # Retrieve relevant context using Pinecone
    relevant_context = retrieve_relevant_context(user_input, context, science_objectives)

    cleaned_context_list = [clean_retrieved_context(chunk) for chunk in relevant_context]
    

    yield "Context Retrieved successfully ✅ ", None, None, None, None, None, None

    keywords = extract_keywords_with_gpt(context, client)

    ads_query = " ".join(keywords)
    
    # Fetch NASA ADS references using the user context
    references = fetch_nasa_ads_references(ads_query)

    yield "ADS references retrieved... ✅ ", None, None, None, None, None, None
    

    yield "🔄 Generating structured response using GPT-4o...", None, None, None, None, None, None
    
    # Generate response from GPT-4
    full_response, response_only = generate_response(
        user_input=user_input,
        science_objectives=science_objectives,  
        relevant_context=relevant_context,
        references=references,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # RAGAS Evaluation
    
    context_ragas = cleaned_context_list
    response_ragas = response_only
    query_ragas = user_input + context
    reference_ragas = "\n\n".join([f"{title}\n{abstract}" for title, abstract, _, _, _, _ in references])

    dataset = []

    dataset.append(
        {
            "user_input":query_ragas,
            "retrieved_contexts":context_ragas,
            "response":response_ragas,
            "reference": "\n\n".join(context_ragas)
        }
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    ragas_evaluation = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), ContextRelevance(), Faithfulness(), ResponseRelevancy(), FactualCorrectness(coverage="low",atomicity="low")],llm=evaluator_llm, embeddings=embeddings)
    
    yield "Response generated successfully ✅ ", None, None, None, None, None, None
    
    # Append user-defined science objectives if provided
    if science_objectives.strip():
        full_response = f"### Science Objectives (User-Defined):\n\n{science_objectives}\n\n" + full_response

    # Export response to Word
    word_doc_path = export_to_word(
        full_response, subdomain, user_input, context, 
        max_tokens, temperature, top_p, frequency_penalty, presence_penalty
    )

    yield "Writing SCDD...Performing RAGAS Evaluation...", None, None, None, None, None, None
    
    # Fetch exoplanet data and generate insights
    exoplanet_data = fetch_exoplanet_data()
    data_insights_uq = generate_data_insights(user_input, client, exoplanet_data)

    # Extract GPT-generated table into DataFrame
    extracted_table_df = gpt_response_to_dataframe(full_response)

    # Combine response and insights
    full_response = f"{full_response}\n\nEnd of Response"

    yield "SCDD produced successfully ✅", None, None, None, None, None, None

    iframe_html = """<iframe width=\"768\" height=\"432\" src=\"https://miro.com/app/live-embed/uXjVKuVTcF8=/?moveToViewport=-331,-462,5434,3063&embedId=710273023721\" frameborder=\"0\" scrolling=\"no\" allow=\"fullscreen; clipboard-read; clipboard-write\" allowfullscreen></iframe>"""
    mapify_button_html = """<a href=\"https://mapify.so/app/new\" target=\"_blank\"><button>Create Mind Map on Mapify</button></a>"""

    yield full_response, relevant_context, ragas_evaluation, extracted_table_df, word_doc_path, iframe_html, mapify_button_html

with gr.Blocks() as demo:
    gr.Markdown("# **The AKSIES Platform [version-0.1]**")

    with gr.Tabs():
        # ===== Tab 1: Original =====
        with gr.Tab("SCDD-GEN"):
            gr.Markdown("## **User Inputs**")
            user_input = gr.Textbox(lines=5, placeholder="Enter your Science Goal...", label="Science Goal")
            context = gr.Textbox(lines=10, placeholder="Enter Context Text...", label="Additional Context")
            subdomain = gr.Textbox(lines=2, placeholder="Define your Subdomain...", label="Subdomain Definition")

            science_objectives_button = gr.Button("User-defined Science Objectives [Optional]")
            science_objectives_input = gr.Textbox(lines=5, placeholder="Enter Science Objectives...", label="Science Objectives", visible=False)
            science_objectives_button.click(lambda: gr.update(visible=True), outputs=[science_objectives_input])

            gr.Markdown("### **Model Parameters**")
            max_tokens = gr.Slider(50, 2000, 150, step=10, label="Max Tokens")
            temperature = gr.Slider(0.0, 1.0, 0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="Top-p")
            frequency_penalty = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="Frequency Penalty")
            presence_penalty = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="Presence Penalty")

            gr.Markdown("## **Model Outputs**")
            full_response = gr.Textbox(label="ExosAI SCDD Generation...")
            relevant_context = gr.Textbox(label="Retrieved Context...")
            ragas_evaluation = gr.Textbox(label="RAGAS Evaluation...")
            extracted_table_df = gr.Dataframe(label="SC Requirements Table")
            word_doc_path = gr.File(label="Download SCDD")
            iframe_html = gr.HTML(label="Miro")
            mapify_button_html = gr.HTML(label="Generate Mind Map on Mapify")

            with gr.Row():
                submit_button = gr.Button("Generate SCDD")
                clear_button = gr.Button("Reset")

            submit_button.click(
                chatbot,
                inputs=[user_input, science_objectives_input, context, subdomain, max_tokens, temperature, top_p, frequency_penalty, presence_penalty],
                outputs=[full_response, relevant_context, ragas_evaluation, extracted_table_df, word_doc_path, iframe_html, mapify_button_html],
                queue=True
            )

            clear_button.click(
                lambda: ("", "", "", "", 150, 0.7, 0.9, 0.5, 0.0, "", "", None, None, None, None, None),
                outputs=[user_input, science_objectives_input, context, subdomain, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, full_response, relevant_context, ragas_evaluation, extracted_table_df, word_doc_path, iframe_html, mapify_button_html]
            )

        # ===== Tab 2: Duplicate for testing =====
        with gr.Tab("SCDD-GEN (Test)"):
            gr.Markdown("## **User Inputs**")
            user_input_2 = gr.Textbox(lines=5, placeholder="Enter your Science Goal...", label="Science Goal")
            context_2 = gr.Textbox(lines=10, placeholder="Enter Context Text...", label="Additional Context")
            subdomain_2 = gr.Textbox(lines=2, placeholder="Define your Subdomain...", label="Subdomain Definition")

            science_objectives_button_2 = gr.Button("User-defined Science Objectives [Optional]")
            science_objectives_input_2 = gr.Textbox(lines=5, placeholder="Enter Science Objectives...", label="Science Objectives", visible=False)
            science_objectives_button_2.click(lambda: gr.update(visible=True), outputs=[science_objectives_input_2])

            gr.Markdown("### **Model Parameters**")
            max_tokens_2 = gr.Slider(50, 2000, 150, step=10, label="Max Tokens")
            temperature_2 = gr.Slider(0.0, 1.0, 0.7, step=0.1, label="Temperature")
            top_p_2 = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="Top-p")
            frequency_penalty_2 = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="Frequency Penalty")
            presence_penalty_2 = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="Presence Penalty")

            gr.Markdown("## **Model Outputs**")
            full_response_2 = gr.Textbox(label="ExosAI SCDD Generation...")
            relevant_context_2 = gr.Textbox(label="Retrieved Context...")
            ragas_evaluation_2 = gr.Textbox(label="RAGAS Evaluation...")
            extracted_table_df_2 = gr.Dataframe(label="SC Requirements Table")
            word_doc_path_2 = gr.File(label="Download SCDD")
            iframe_html_2 = gr.HTML(label="Miro")
            mapify_button_html_2 = gr.HTML(label="Generate Mind Map on Mapify")

            with gr.Row():
                submit_button_2 = gr.Button("Generate SCDD")
                clear_button_2 = gr.Button("Reset")

            submit_button_2.click(
                chatbot,
                inputs=[user_input_2, science_objectives_input_2, context_2, subdomain_2, max_tokens_2, temperature_2, top_p_2, frequency_penalty_2, presence_penalty_2],
                outputs=[full_response_2, relevant_context_2, ragas_evaluation_2, extracted_table_df_2, word_doc_path_2, iframe_html_2, mapify_button_html_2],
                queue=True
            )

            clear_button_2.click(
                lambda: ("", "", "", "", 150, 0.7, 0.9, 0.5, 0.0, "", "", None, None, None, None, None),
                outputs=[user_input_2, science_objectives_input_2, context_2, subdomain_2, max_tokens_2, temperature_2, top_p_2, frequency_penalty_2, presence_penalty_2, full_response_2, relevant_context_2, ragas_evaluation_2, extracted_table_df_2, word_doc_path_2, iframe_html_2, mapify_button_html_2]
            )

demo.launch(share=True)

