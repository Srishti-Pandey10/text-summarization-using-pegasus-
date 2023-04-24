import streamlit as st
import joblib,torch
import time
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_tokenizer = joblib.load("finalized_tokenizer.sav")
loaded_model = joblib.load("finalized_model.sav")

st.title('Text Summarization using Pegasus')

txt = st.text_area('Enter Text to summarize here', '')

with st.sidebar:
    st.subheader("Text Summarization using Pegasus")
    
    st.write("PEGASUS uses an encoder-decoder model for sequence-to-sequence learning. In such a model, the encoder will first take into consideration the context of the whole input text and encode the input text into something called context vector, which is basically a numerical representation of the input text. This numerical representation will then be fed to the decoder whose job is decode the context vector to produce the summary.")
    image =Image.open("Pegasus_model.png")
    
    st.image(image, caption='Pegasus Model')
    st.code("App built by Srishti Pandey",language="python")


if st.button('Summarize'):
    with st.spinner('Summarizing..'):
        batch = loaded_tokenizer(txt, truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = loaded_model.generate(**batch)
        tgt_text = loaded_tokenizer.batch_decode(translated, skip_special_tokens=True)
    st.success('Summarized Text')
    st.subheader(tgt_text[0])


