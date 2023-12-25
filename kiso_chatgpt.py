import streamlit as st
import os

from langchain.chains import  ConversationChain
from langchain.chat_models import ChatOpenAI

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("KiSo - Chat") 
st.divider()

with st.sidebar:
    st.title("KiSo - Chat")
    apiKey=st.text_input("OpenAI API Key")
    st.divider()
    st.empty()
    model = st.selectbox("Choose a model:", ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"])
    st.divider()
    st.empty()
    if st.button("New Chat :page_facing_up:"):
        st.session_state.messages = []

# The code block you provided is setting up the OpenAI API key, creating an instance of the ChatOpenAI
# model, and initializing a ConversationChain.
try:
    os.environ['OPENAI_API_KEY'] = apiKey 
    llm = ChatOpenAI(
        model=model,
        temperature=1,
        max_tokens=250,  # Angenommene Anzahl an Tokens
        top_p=1.0,
    )
    
    inhalt_chain = ConversationChain(llm=llm, verbose=True)

# The code block you provided is responsible for displaying the chat messages from the history on app
# rerun and allowing the user to input their question.
# Display chat messages from history on app rerun

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt= st.chat_input("Ask your question!")

# This code block is responsible for handling the user's question and generating a response from the
# assistant.
    if prompt:
        with st.chat_message("user"):   
            st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = inhalt_chain.predict(input=prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
except Exception as e:
    st.write(st.write(str(e)))
           