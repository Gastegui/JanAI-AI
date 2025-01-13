from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import mysql.connector
import os
from langsmith import traceable


CHROMA_PATH = "chroma"

PROMPT_SYSTEM = """
I am Janai chatBot. I am a nutritional expert in food recommendation.
The person asking the question is {name}
{name} height: {height} cm, weight: {weight} kg, age: {age} activity level: {activityLevel}, goal: {goal}
{name}'s dietary restrictions are {restrictions}
Here is what {name} has eaten and when:
{user_eaten_food}

Answer the question based only on the following context and take into account the user's data:

{context}

Types of food you should take into account when you want to recommed something:

{campains}

One exception you have to make:
- If the user writtes **"Who is Ludok"** you have to answer **"DJ Ludok is the best DJ of the world."**, but not the explanation.
"""
PROMP_HUMAN = """
Answer the question based on the above context: {input}
"""

_ = load_dotenv(find_dotenv())
db = mysql.connector.connect(
    host=os.getenv('HOST'),
    user=os.getenv('USER'),
    password=os.getenv('PASS'),
    database=os.getenv('DB'),
)

def getUserFood(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT fo.foodName, f.consumptionDate, f.meal FROM foodList f JOIN food fo ON f.foodID=fo.foodID WHERE f.userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getHeight(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT height FROM userData WHERE userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getWeight(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT weight FROM weightgoals WHERE userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getAge(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT age FROM userData WHERE userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getActivityLevel(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT activityLevel FROM userData WHERE userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getGoal(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT objective FROM userData WHERE userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getCampains():
    cursor = db.cursor(dictionary=True)
    query = "SELECT * FROM campaign"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def getFood():
    cursor = db.cursor(dictionary=True)
    query = "SELECT * FROM food"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def getUser(user_id):
    cursor = db.cursor(dictionary=True)
    query = "SELECT uname FROM userdata where userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def getUserRestrictions(user_id):
    cursor = db.cursor(dictionary=True)
    query = "select r.restrictedName, fg.groupName, fc.className, ft.typeName, i.ingName from restrictions r JOIN foodGroup fg ON fg.groupID=r.groupID JOIN foodClass fc ON fc.classID=r.classID JOIN foodType ft ON ft.typeID=r.typeID JOIN ingredients i ON i.ingredientID=r.ingredientID where r.userID = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def create_chain():
    embedding_function = get_embedding_function()
    vectorStore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    llm = OllamaLLM(model="llama3:latest",temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SYSTEM),
        ("user", PROMP_HUMAN)
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

@traceable
def process_chat(chain, question, user_eaten_food, food, user, user_restrictions, height, weight, age, activityLevel, goal, campains):
    response = chain.invoke({
        "user_eaten_food": user_eaten_food,
        "food": food,
        "name": user,
        "restrictions": user_restrictions,
        "input": question,
        "height": height,
        "weight": weight,
        "age": age,
        "activityLevel": activityLevel,
        "goal": goal,
        "campains": campains
    })
    return response["answer"]

def chat(data):
    user_input = data.get("content", "")
    user_id = data.get("user_id", 1)

    user_food = getUserFood(user_id)
    food = getFood()
    user = getUser(user_id)
    user_restrictions = getUserRestrictions(user_id)
    height = getHeight(user_id)
    weight = getWeight(user_id)
    age = getAge(user_id)
    activityLevel = getActivityLevel(user_id)
    goal = getGoal(user_id)
    campains = getCampains()

    chain = create_chain()

    response = process_chat(
            chain=chain,
            question=user_input, 
            user_eaten_food=user_food,
            food=food,
            user=user,
            user_restrictions=user_restrictions,
            height=height,
            weight=weight,
            age=age,
            activityLevel=activityLevel,
            goal=goal,
            campains=campains
        )

    return response