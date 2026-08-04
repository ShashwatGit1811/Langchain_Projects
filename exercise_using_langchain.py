import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

MODEL  = "llama-3.1-8b-instant"


# while True:
# HumanMessage=input("You : ")
# print(prompt)

    # if HumanMessage in ["Bye", "Nikal", "Milte Hai"]:
    #     break

    #Method 1 :
    # llm = init_chat_model(MODEL,model_provider="groq")

    #Method 2:
llm=ChatGroq(model=MODEL)

questions=[
        "What is FastApi?",
        "What is Langchain?",
        "What is MySql?"
    ]

    # print(llm)
    #for Recive and Store in Responce From server Or Model
for q in questions:
        print("Questions : ",q)
        messages=[
            SystemMessage(content="Give me answer in 2-3 lines only."),
            HumanMessage(content=q)
        ]

        response = llm.invoke(messages)
        print("Response : ",response.content)
        print("Tokens",response.usage_metadata)
        # # print("Time",response.time)

        print("\n\n")
    
    