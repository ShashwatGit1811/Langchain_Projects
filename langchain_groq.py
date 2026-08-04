import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage

# loading env files
load_dotenv()

# Model declaration
MODEL  = "llama-3.1-8b-instant"

while True:
    prompt=input("You : ")
    # print(prompt)

    #Method 1 :
    # llm = init_chat_model(MODEL,model_provider="groq")

    #Method 2:
    llm=ChatGroq(model=MODEL)
    
    # print(llm)
    #Recieving and Storing the response from model
    response = llm.invoke(prompt)

    # Printing Output    
    print(response)
    print("Groq : ",response.content)
    print(response.usage_metadata)
    
