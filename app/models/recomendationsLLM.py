import os

import mysql.connector
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        password=os.getenv('PASS'),
        database=os.getenv('DB'),
    )


CHROMA_PATH = 'chroma'

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


def getUserFood(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT fo.foodName, f.consumptionDate, f.meal FROM foodList f JOIN food fo ON f.foodID=fo.foodID JOIN userData u ON f.userID=u.userID WHERE u.username = %s AND f.consumptionDate >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getHeight(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT height FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getWeight(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT weight FROM weightgoals w JOIN userData u ON w.userID=u.userID WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getAge(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT age FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getActivityLevel(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT activityLevel FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getGoal(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT objective FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getCampains(db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT * FROM campaign'
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


def getFood(db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT * FROM food'
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


def getUser(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT uname FROM userdata where username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def getUserRestrictions(username, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT r.restrictedName, fg.groupName, fc.className, ft.typeName, i.ingName from restrictions r JOIN foodGroup fg ON fg.groupID=r.groupID JOIN foodClass fc ON fc.classID=r.classID JOIN foodType ft ON ft.typeID=r.typeID JOIN ingredients i ON i.ingredientID=r.ingredientID JOIN userData u ON r.userID=u.userID where u.username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_embedding_function():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings


def create_chain():
    embedding_function = get_embedding_function()
    vectorStore = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    llm = OllamaLLM(model='llama3:latest', temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [('system', PROMPT_SYSTEM), ('user', PROMP_HUMAN)]
    )

    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = vectorStore.as_retriever(search_kwargs={'k': 5})

    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain


def process_chat(
    chain,
    question,
    user_eaten_food,
    food,
    user,
    user_restrictions,
    height,
    weight,
    age,
    activityLevel,
    goal,
    campains,
):
    response = chain.invoke(
        {
            'user_eaten_food': user_eaten_food,
            'food': food,
            'name': user,
            'restrictions': user_restrictions,
            'input': question,
            'height': height,
            'weight': weight,
            'age': age,
            'activityLevel': activityLevel,
            'goal': goal,
            'campains': campains,
        }
    )
    return response['answer']


def chat(data):
    with get_db_connection() as db:
        user_input = data.get('content', '')
        username = data.get('username', '')

        print('User input: ', user_input)
        print('Username: ', username)

        user_food = getUserFood(username=username, db=db)
        food = getFood(db=db)
        user = getUser(username=username, db=db)
        user_restrictions = getUserRestrictions(username=username, db=db)
        height = getHeight(username=username, db=db)
        weight = getWeight(username=username, db=db)
        age = getAge(username=username, db=db)
        activityLevel = getActivityLevel(username=username, db=db)
        goal = getGoal(username=username, db=db)
        campains = getCampains(db=db)

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
            campains=campains,
        )

    return response