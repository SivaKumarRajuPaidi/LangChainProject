## Integrate our code OpenAI API
import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



import streamlit as st

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

# streamlit framework

st.title('Langchain Demo With OPENAI API')
input_text=st.text_input("Search the topic u want")

## OPENAI LLMS
llm=OpenAI(temperature=0.8)


prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")



chain = LLMChain(llm=llm, prompt=prompt)
#chain.run("colorful socks")

if input_text:
    #st.write(llm.predict(prompt.format(product=input_text)))
    st.write(chain.run(input_text))