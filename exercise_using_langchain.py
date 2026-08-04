# AI RESPONSES WITH STATIC LIST OF QUESTIONS 

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

# model declaration
MODEL  = "llama-3.1-8b-instant"

llm=ChatGroq(model=MODEL)

questions=[
        "What is FastApi?",
        "What is Langchain?",
        "What is MySql?"
    ]

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
    
    
