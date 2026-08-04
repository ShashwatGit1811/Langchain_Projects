# CHAT WITH TEMPORARY RUNTIME MEMORY 

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

# Model declaration
MODEL  = "llama-3.1-8b-instant"

messages=[]

while True:
    prompt=input("You : ")
    # print(prompt)
    
    # Method 1 :
    # llm = init_chat_model(MODEL,model_provider="groq")

    # Method 2:
    llm=ChatGroq(model=MODEL)

    message=[
        SystemMessage(content="You have to give all the answers in 2 to 3 lines with one real life examples."),
        HumanMessage(content=prompt)
    ]

    response=llm.invoke(message)

    messages.append({"role" : "user", "content" : prompt})
    messages.append({"role" : "assistant", "content" : response.content})
    
    print("Response : ",response.content)
    print("Meta Data : ",response.usage_metadata)

    
