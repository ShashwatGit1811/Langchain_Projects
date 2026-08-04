import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

MODEL  = "llama-3.1-8b-instant"

while True:
    prompt=input("You : ")
    
    # print(prompt)

    if prompt in ["Bye", "Nikal", "Milte Hai"]:
        break

    #Method 1 :
    # llm = init_chat_model(MODEL,model_provider="groq")

    #Method 2:
    llm=ChatGroq(model=MODEL)

    # print(llm)
    #for Recive and Store in    Responce From server Or Model
    response = llm.invoke(prompt)

    #for Taking Perticualr Output 
    # print("Groq : ")
    print(response)
    # print("Groq : ",response.content)
    # print(response.usage_metadata)

