'''
Memory using Conversation Buffer Window memory which stores last k numbers of 
prompts(System,Ai ans,Human prompts)
'''


import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.memory import ConversationBufferWindowMemory

load_dotenv()

MODEL  = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are an expert IT professional.
You know everyhing about IT field and you are master of it.
You have to be helpful and give concise answers.
"""

llm=ChatGroq(
    model= MODEL,
    temperature=0.3,
    max_retries=3
)



# TASK 1: Create memory
memory=ConversationBufferWindowMemory(k=10,return_messages=True)


# TASK 2: Build prompt
prompt=ChatPromptTemplate.from_messages([
    ("system",SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

parser = StrOutputParser()


# TASK 3: Build chain
chain=prompt|llm|parser

# TASKS 4 & 5: Chat loop
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
    

