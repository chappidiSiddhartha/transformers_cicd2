import streamlit as st
from transformers import pipeline, AutoTokenizer
from azureml.core import Workspace, Experiment, Run 
import os
import json

# Azure ML Setup
try:
    # Connect to Azure ML Workspace
    ws = Workspace.from_config()  # Ensure a config.json file exists in your directory
    experiment_name = "huggingface_streamlit"
    experiment = Experiment(ws, experiment_name)
    run = experiment.start_logging()
    st.sidebar.success("Connected to Azure ML Workspace")
except Exception as e:
    st.sidebar.error(f"Azure ML Setup Failed: {e}")
    run = None

# Streamlit App Title
st.title('Hugging Face Transformers with Streamlit and Azure ML')

# Sidebar for Task Selection
model_type = st.sidebar.selectbox(
    "Select a Task", 
    ("Sentiment Analysis", "Text Generation", "Named Entity Recognition")
)

# Load the appropriate model based on the selected task
if model_type == "Sentiment Analysis":
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = pipeline("sentiment-analysis", model=model_name)
elif model_type == "Text Generation":
    model_name = "gpt2"
    model = pipeline("text-generation", model=model_name)
elif model_type == "Named Entity Recognition":
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    model = pipeline("ner", model=model_name)

# Save the model and tokenizer for reproducibility
model_dir = 'outputs/models/'
os.makedirs(model_dir, exist_ok=True)
model.model.save_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_dir)

# Display the selected task
st.write(f"### Selected Task: {model_type}")

# Text input for user
user_input = st.text_area('Enter your text here:', height=150)

# Button to trigger processing
button = st.button("Analyze Text")

# Process the input text when button is clicked
if button:
    if user_input:
        with st.spinner("Processing... Please wait."):
            try:
                if model_type == "Sentiment Analysis":
                    result = model(user_input)
                    st.write(f"### Sentiment: {result[0]['label']}")
                    st.write(f"**Confidence Score**: {result[0]['score']:.4f}")
                    
                    # Log results to Azure ML
                    if run:
                        run.log("Sentiment", result[0]['label'])
                        run.log("Confidence", result[0]['score'])

                elif model_type == "Text Generation":
                    result = model(user_input, max_length=50, num_return_sequences=1)
                    st.write(f"### Generated Text:")
                    st.write(result[0]['generated_text'])
                    
                    # Log results to Azure ML
                    if run:
                        run.log("Generated Text", result[0]['generated_text'])

                elif model_type == "Named Entity Recognition":
                    result = model(user_input)
                    st.write("### Named Entities Found:")
                    for entity in result:
                        st.write(f"- **Entity**: {entity['word']} | **Label**: {entity['entity']} | **Score**: {entity['score']:.4f}")
                    
                    # Log results to Azure ML
                    if run:
                        entities_json = json.dumps(result)
                        run.log("NER Results", entities_json)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text for analysis.")

# Finalize Azure ML Run
if run:
    run.complete()
    st.sidebar.success("Experiment logged in Azure ML")

# Footer
st.markdown(
    """
    ---
    This app is powered by [Hugging Face Transformers](https://huggingface.co/transformers/) and [Streamlit](https://streamlit.io/), with experiments logged to [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/).
    """
)
