#  DB functions
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage
from llm_and_chains import eval_chain

def get_connection():
    conn = sqlite3.connect("chatdb.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn=get_connection()
    cursor=conn.cursor()

    cursor.executescript("""
            create table if not exists chat_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id integer DEFAULT 0,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                model_used TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')));       
                
            CREATE TABLE IF NOT EXISTS eval_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id integer DEFAULT 0,
                chat_log_id INTEGER NOT NULL,
                relevance REAL NOT NULL,
                coherence REAL NOT NULL,
                conciseness REAL NOT NULL,
                overall REAL NOT NULL,
                feedback TEXT NOT NULL,
                created_at TEXT DEFAULT(datetime('now')),
                FOREIGN KEY(chat_log_id) REFERENCES chat_logs(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_memory(
                id INTEGER PRIMARY KEY AUTOINCREMENT ,
                session_id integer DEFAULT 0,
                role text not null,
                content text not null,
                created_at TEXT DEFAULT (datetime('now')));
        """)
    conn.commit()
    conn.close()
    print("\nDatabase Ready\n")


def save_chat(session_id,chat):
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("""INSERT INTO chat_logs
           (session_id, user_message, ai_response, model_used,
            input_tokens, output_tokens)
           VALUES (?, ?, ?, ?, ?, ?)""",
           (
           session_id,
           chat["user_message"],
           chat["ai_response"],
           chat["model"],
           chat["input_tokens"],
           chat["output_tokens"]))

    conn.commit()
    chat_id = cursor.lastrowid 
    conn.close()

    return chat_id


def save_eval(chat_id,scores,session_id):
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("""INSERT INTO eval_logs
           (session_id,chat_log_id,relevance, coherence, conciseness, overall,
            feedback)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
           (
           session_id,
           chat_id,
           scores["relevance"],
           scores["coherence"],
           scores["conciseness"],
           scores["overall"],
           scores["feedback"]))

    conn.commit()
    conn.close()


def save_message(session_id,role,prompt):
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("""INSERT INTO conversation_memory
           (session_id, role, content)
           VALUES (?, ?, ?)""",
           (session_id,role,prompt)
        )

    conn.commit()
    conn.close()


def load_history(session_id)-> list:
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute("select role,content from conversation_memory") 
    '''where session_id=? order by id",(session_id,)'''

    rows=cursor.fetchall()
    conn.commit()
    conn.close()

    messages=[]
    for row in rows:
        if row[0]=="user":
            messages.append(HumanMessage(content=row[1]))
        else:
            messages.append(AIMessage(content=row[1]))

    return messages


def evaluate(chat_id,question: str, answer: str,session_id) -> float:
    try:
        scores  = eval_chain.invoke({"question": question, "answer": answer})
        overall = round((
            scores.get("relevance",   0) +
            scores.get("coherence",   0) +
            scores.get("conciseness", 0)
        ) / 3, 2)
        scores["overall"] = overall

        save_eval(chat_id,scores,session_id)
        return scores["overall"]


    except Exception as e:
        print(f"Eval failed: {e}")
        return 0.0


def new_session_id():
    # calculation current session_id 
    conn=get_connection()
    cursor=conn.cursor()
    # session_id 
    cursor.execute("select MAX(session_id) from conversation_memory;")
    max_id=cursor.fetchone()
    if max_id[0]:
        session_id=max_id[0]+1
    else:
        session_id=1
    conn.commit()
    conn.close()
    
    return session_id  


def get_sessions():
    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute("""
    SELECT cm.session_id,
    substr(cm.content,1,40) AS title 
    FROM conversation_memory cm
    WHERE cm.role = 'user' 
    AND cm.id = (
                SELECT MIN(id)
                FROM conversation_memory
                WHERE session_id = cm.session_id
                AND role = 'user'
                )
    ORDER BY cm.session_id DESC;
    """)

    sessions=cursor.fetchall()

    conn.commit()
    conn.close()

    return sessions


def get_session_history(session_id):
    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute("select role,content from conversation_memory where session_id=?",(session_id,))

    session_data=cursor.fetchall()

    conn.commit()
    conn.close()

    return session_data


def delete_session_history(session_id):
    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute("Delete from conversation_memory where session_id=?",(session_id,))
    cursor.execute("Delete from chat_logs where session_id=?",(session_id,))
    cursor.execute("Delete from eval_logs where session_id=?",(session_id,))


    conn.commit()
    conn.close()
