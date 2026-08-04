'''
Memory using Conversation Buffer Window memory which stores last k numbers of 
prompts(System,Ai ans,Human prompts)
'''


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.memory import ConversationBufferWindowMemory

# loading env files
load_dotenv()

# model defination
MODEL  = "llama-3.1-8b-instant"

# system prompt defination for llm
SYSTEM_PROMPT = """
You are an expert IT professional.
You know everyhing about IT field and you are master of it.
You have to be helpful and give concise answers.
"""

# defination of llm
llm=ChatGroq(
    model= MODEL,
    temperature=0.3,
    max_retries=3
)


# Creating memory
memory=ConversationBufferWindowMemory(k=10,return_messages=True)


# Build prompt
prompt=ChatPromptTemplate.from_messages([
    ("system",SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

parser = StrOutputParser()


# Build chain
chain=prompt|llm|parser

# chat bot infinte loop
print("Type exit to quit")

while True:

    user_input=input("\nYou : ").strip()
    if user_input=="exit":
        print("GoodBye!")
        break

    history=memory.load_memory_variables({})["history"]
    reply=chain.invoke({"input":user_input,"history" : history})
    memory.save_context({"input":user_input},{"output":reply})

    print(f"Ai : {reply}")
