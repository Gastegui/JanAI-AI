"""
Module for interfacing with Ollama LLM for the purposes of providing chatbot functionality.
"""

import os

import mysql.connector
from dotenv import find_dotenv, load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def get_db_connection():
    """
    Establishes a connection to the MySQL database using credentials from environment variables.

    Returns:
        mysql.connector.connect: A connection object to the MySQL database.
    """
    return mysql.connector.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        password=os.getenv('PASS'),
        database=os.getenv('DB'),
    )


CHROMA_PATH = 'chroma'

PROMPT_SYSTEM = """
# Role and Identity
You are Janai, an AI nutrition assistant specializing in personalized dietary recommendations. You provide evidence-based nutrition advice while maintaining a supportive and encouraging tone.

# User Profile
Personal Information:
- Name: {name}
- Height: {height} cm
- Weight: {weight} kg
- Age: {age}
- Activity Level: {activityLevel}
- Health Goal: {goal}

Dietary Considerations:
- Restrictions: {restrictions}
- Recent Meals: {user_eaten_food}

# Behavioral Guidelines
1. Never recommend foods containing listed restrictions
2. Consider recent meals when making suggestions to ensure variety
3. Tailor portions and caloric recommendations to activity level
4. Always account for stated health goals in recommendations
5. Base all advice on provided context and scientific evidence
6. Provide brief explanations for nutritional recommendations

# Response Format
- Keep responses concise and actionable
- Structure recommendations in clear sections
- Include approximate nutritional values when relevant
- Suggest practical alternatives for restricted foods

Context for recommendations:
{context}
"""

PROMP_HUMAN = """
Based on your nutrition expertise and the above context, please address: {input}
"""

_ = load_dotenv(find_dotenv())


def get_user_food(username, db):
    """
    Retrieves the list of foods consumed by the user in the past 7 days.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list of dictionaries containing food names, consumption dates, and meal types.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT fo.foodName, f.consumptionDate, f.meal FROM foodList f JOIN food fo ON f.foodID=fo.foodID JOIN userData u ON f.userID=u.userID WHERE u.username = %s AND f.consumptionDate >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_height(username, db):
    """
    Retrieves the user's height from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's height.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT height FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_weight(username, db):
    """
    Retrieves the user's weight from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's weight.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT weight FROM weightgoals w JOIN userData u ON w.userID=u.userID WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_age(username, db):
    """
    Retrieves the user's age from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's age.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT age FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_activity_level(username, db):
    """
    Retrieves the user's activity level from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's activity level.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT activityLevel FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_goal(username, db):
    """
    Retrieves the user's health goal from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's health goal (e.g., fat loss, muscle gain).
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT objective FROM userData WHERE username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result

def get_user(username, db):
    """
    Retrieves the user's unique username from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list containing the user's unique username.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT uname FROM userdata where username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_user_restrictions(username, db):
    """
    Retrieves the user's food restrictions based on their profile from the database.

    Args:
        username (str): The username of the user.
        db (mysql.connector.connection): The MySQL database connection object.

    Returns:
        list: A list of restrictions, including food groups, classes, types, and ingredients.
    """
    cursor = db.cursor(dictionary=True)
    query = 'SELECT fg.groupName, fc.className, ft.typeName, i.ingName from restrictions r JOIN foodGroup fg ON fg.groupID=r.groupID JOIN foodClass fc ON fc.classID=r.classID JOIN foodType ft ON ft.typeID=r.typeID JOIN ingredients i ON i.ingredientID=r.ingredientID JOIN userData u ON r.userID=u.userID where u.username = %s'
    cursor.execute(query, (username,))
    result = cursor.fetchall()
    cursor.close()
    return result


def get_embedding_function():
    """
    Initializes the embedding function using Ollama embeddings for text processing.

    Returns:
        OllamaEmbeddings: The embeddings object for use in vectorization and retrieval.
    """
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings


def create_chain():
    """
    Creates a retrieval chain combining the embedding function, vector store, and LLM for chat-based nutrition advice.

    Returns:
        langchain.chains.RetrievalChain: A chain used for processing and generating chatbot responses.
    """
    embedding_function = get_embedding_function()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )
    #print(f"Number of documents in the vector store: {len(vector_store._collection.count()}")

    llm = OllamaLLM(model='llama3:latest', temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [('system', PROMPT_SYSTEM), ('user', PROMP_HUMAN)]
    )

    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain, retriever


def process_chat(
    chain,
    question,
    user_eaten_food,
    user,
    user_restrictions,
    height,
    weight,
    age,
    activity_level,
    goal,
    context,
):
    """
    Processes the user input through the retrieval chain to generate a nutrition-based response.

    Args:
        chain (langchain.chains.RetrievalChain): The retrieval chain used to generate responses.
        question (str): The user's nutrition-related question.
        user_eaten_food (list): List of foods the user has consumed recently.
        food (list): A list of all available food items in the database.
        user (str): The user's name.
        user_restrictions (list): The list of foods the user is restricted from consuming.
        height (list): The user's height.
        weight (list): The user's weight.
        age (list): The user's age.
        activity_level (list): The user's activity level.
        goal (list): The user's health goal.

    Returns:
        str: The response generated by the AI nutrition assistant.
    """
    response = chain.invoke(
        {
            'user_eaten_food': user_eaten_food,
            'name': user,
            'restrictions': user_restrictions,
            'input': question,
            'height': height,
            'weight': weight,
            'age': age,
            'activityLevel': activity_level,
            'goal': goal,
            'context': context,
        }
    )
    return response['answer']

def chat(data):
    """
    Main function to handle chat interaction. Retrieves necessary data and processes the input through the chain.

    Args:
        data (dict): A dictionary containing the user's question and username.

    Returns:
        str: The response generated by the AI nutrition assistant.
    """
    with get_db_connection() as db:
        user_input = data.get('content', '')
        username = data.get('username', '')

        print('Question: ', user_input)
        print('Username: ', username)

        user_food =  get_user_food(username=username, db=db)
        user = get_user(username=username, db=db)
        user_restrictions = get_user_restrictions(username=username, db=db)
        height = get_height(username=username, db=db)
        weight = get_weight(username=username, db=db)
        age = get_age(username=username, db=db)
        activity_level = get_activity_level(username=username, db=db)
        goal = get_goal(username=username, db=db)

        chain, retriever = create_chain()
        retrieved_documents = retriever.get_relevant_documents(user_input)
        #print('Retrieved documents: ', retrieved_documents)

        context = "\n".join(doc.page_content for doc in retrieved_documents)
        #print('Context: ', context)

        response = process_chat(
            chain=chain,
            question=user_input,
            user_eaten_food=user_food,
            user=user,
            user_restrictions=user_restrictions,
            height=height,
            weight=weight,
            age=age,
            activity_level=activity_level,
            goal=goal,
            context=context,
        )

    return response
