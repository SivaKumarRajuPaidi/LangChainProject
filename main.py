
import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Streamlit Framework

st.title('Langchain demo with OpenAI')
input_text = st.text_input("Which joke would you like to hear?")

#LLM
llm = OpenAI(temperature=0.9)

#first_prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

#Prompt Templates
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a story title about {topic}?",
)
script_prompt = PromptTemplate(
    input_variables=["title"],
    template="Tell me a short story based on this TITLE: {title}",
)


title_chain = LLMChain(llm=llm, prompt=title_prompt,verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_prompt, output_key='script')

sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'],output_variables=['title','script'], verbose =True)

if input_text:
    #st.write(llm.predict(prompt.format(product=input_text)))
    response = sequential_chain.run({'topic':input_text})
    st.write(response['title'])
    st.write(response['script'])