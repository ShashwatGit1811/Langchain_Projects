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
    
    # Method 1 :
    # llm = init_chat_model(MODEL,model_provider="groq")

    # Method 2:
    llm=ChatGroq(model=MODEL)

    message=[
        SystemMessage(content="You have to give all the answers in 2 to 3 lines with one real life examples."),
        HumanMessage(content=prompt)
    ]

    response=llm.invoke(message)

    print("Response : ",response.content)
    print("Meta Data : ",response.usage_metadata)
    if prompt in ["Bye", "Nikal", "Milte Hai"]:
        break

    # print(llm)
    #for Recive and Store in Responce From server Or Model
