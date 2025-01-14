"""
Module for interfacing with Ollama LLM for the purposes of predicting calorie intake
"""
import os

import mysql.connector
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        password=os.getenv('PASS'),
        database=os.getenv('DB'),
    )


llm = OllamaLLM(model='llama3:latest', temperature=0)

prompt_template = """
You are a highly skilled nutrition expert. Based on the following data:
- Height: {height} cm
- Weight: {weight} kg
- Age: {age} years
- Waist: {waist} cm
- Neck: {neck} cm
- Hips: {hips} cm (optional for males)
- Gender: {gender}
- Goal Weight: {goalWeight} kg
- Timeframe to Achieve Goal Weight: {durationToAchieveGoalWeight} weeks
- Activity Level: {activityLevel} (e.g., Sedentary, Lightly Active, Moderately Active, Very Active)
- Objective (e.g., fat loss, muscle gain, maintenance): {objective}
- Basal Metabolic Rate (BMR) calculations:
    - Mifflin-St Jeor: {bmrMifflin}
    - Harris-Benedict: {bmrHarrisBenedict}
    - Katch-McArdle: {bmrKatchMcArdle}
- Total Daily Energy Expenditure (TDEE) calculations:
    - Mifflin-St Jeor: {tdeeMifflin}
    - Harris-Benedict: {tdeeHarrisBenedict}
    - Katch-McArdle: {tdeeKatchMcArdle}
- Body Fat Percentage: {bodyFat}% 
- Total Weight Loss Target: {totalWeightLoss} kg
- Weekly Caloric Deficit Target: {weeklyDeficit} kcal
- Daily Calorie Intake Recommendations:
    - Mifflin-St Jeor: {dailyCalorieIntakeMifflin} kcal/day
    - Harris-Benedict: {dailyCalorieIntakeHarrisBenedict} kcal/day
    - Katch-McArdle: {dailyCalorieIntakeKatchMcArdle} kcal/day

Using the above information, compare the three daily calorie intake recommendations (Mifflin-St Jeor, Harris-Benedict, Katch-McArdle). Follow these steps:
1. Identify the most suitable recommendation based on the individual's activity level, body fat percentage, and objective (e.g., fat loss, muscle gain, or maintenance).
2. Consider which method aligns best with the goal timeframe and calorie deficit target.
3. Select the most balanced option for sustainability and health.

Provide only the value of the chosen Daily Calorie Intake (e.g., 2000.00), with no additional explanation or text.
"""

prompt = PromptTemplate(
    input_variables=[
        'height',
        'weight',
        'age',
        'waist',
        'neck',
        'hips',
        'gender',
        'goalWeight',
        'durationToAchieveGoalWeight',
        'activityLevel',
        'objective',
        'bmrMifflin',
        'bmrHarrisBenedict',
        'bmrKatchMcArdle',
        'tdeeMifflin',
        'tdeeHarrisBenedict',
        'tdeeKatchMcArdle',
        'bodyFat',
        'totalWeightLoss',
        'weeklyDeficit',
        'dailyCalorieIntakeMifflin',
        'dailyCalorieIntakeHarrisBenedict',
        'dailyCalorieIntakeKatchMcArdle',
    ],
    template=prompt_template,
)

chain = LLMChain(llm=llm, prompt=prompt)


def get_user_data(user_id, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT height, age, waist, neck, hips, gender, activityLevel, objective, bmrMifflin, bmrHarrisBenedict, bmrKatchMcArdle, tdeeMifflin, tdeeHarrisBenedict, tdeeKatchMcArdle, bodyFat, totalWeightLoss, weeklyDeficit, dailyCalorieIntakeMifflin, dailyCalorieIntakeHarrisBenedict, dailyCalorieIntakeKatchMcArdle FROM userData WHERE userID = %s;'
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    cursor.close()
    return result


def get_weight_data(user_id, db):
    cursor = db.cursor(dictionary=True)
    query = 'SELECT * FROM weightGoals WHERE userID = %s ORDER BY registerDate DESC LIMIT 1;'
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    cursor.close()
    return result


def calculate_calories(user_id: int):
    with get_db_connection() as db:
        user_data = get_user_data(user_id=user_id, db=db)
        weight_goals = get_weight_data(user_id=user_id, db=db)

        if not user_data:
            return ({'error': f'No user found with ID {user_id}'}, 404)

        response = chain.invoke(
            {
                'height': user_data['height'],
                'weight': weight_goals['weight'],
                'age': user_data['age'],
                'waist': user_data['waist'],
                'neck': user_data['neck'],
                'hips': user_data['hips'],
                'gender': user_data['gender'],
                'goalWeight': weight_goals['goalWeight'],
                'durationToAchieveGoalWeight': weight_goals[
                    'durationToAchieveGoalWeight'
                ],
                'activityLevel': user_data['activityLevel'],
                'objective': user_data['objective'],
                'bmrMifflin': user_data['bmrMifflin'],
                'bmrHarrisBenedict': user_data['bmrHarrisBenedict'],
                'bmrKatchMcArdle': user_data['bmrKatchMcArdle'],
                'tdeeMifflin': user_data['tdeeMifflin'],
                'tdeeHarrisBenedict': user_data['tdeeHarrisBenedict'],
                'tdeeKatchMcArdle': user_data['tdeeKatchMcArdle'],
                'bodyFat': user_data['bodyFat'],
                'totalWeightLoss': user_data['totalWeightLoss'],
                'weeklyDeficit': user_data['weeklyDeficit'],
                'dailyCalorieIntakeMifflin': user_data[
                    'dailyCalorieIntakeMifflin'
                ],
                'dailyCalorieIntakeHarrisBenedict': user_data[
                    'dailyCalorieIntakeHarrisBenedict'
                ],
                'dailyCalorieIntakeKatchMcArdle': user_data[
                    'dailyCalorieIntakeKatchMcArdle'
                ],
            }
        )
    return response['text']
