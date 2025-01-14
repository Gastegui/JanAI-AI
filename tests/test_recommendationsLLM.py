"""Recommendation LLM test module"""
from unittest.mock import MagicMock, patch

from app.models.recomendationsLLM import chat


@patch('app.models.recomendationsLLM.mysql.connector.connect')
@patch('app.models.recomendationsLLM.OllamaEmbeddings')
@patch('app.models.recomendationsLLM.Chroma')
@patch('app.models.recomendationsLLM.OllamaLLM')
@patch('app.models.recomendationsLLM.ChatPromptTemplate.from_messages')
def test_chat(
    mock_chat_prompt,
    mock_ollama_llm,
    mock_chroma,
    mock_ollama_embeddings,
    mock_connector,
):
    """
    Test case for the inner functioning of recommendation chat script.

    Args:
        mock_chat_prompt (MagicMock): Mock for the LLM prompt.
        mock_ollama_llm (MagicMock): Mock for the Ollama LLM model.
        mock_chroma (MagicMock): Mock for the Chroma db object.
        mock_ollama_embeddings (MagicMock): Mock for the Ollama LLM's Embedding object.
        mock_connector (MagicMock): Mock for the MySQL db connection.

    Asserts:
        - 200 status code.
        - Proper return of chatbot.
    """
    # Mock the database connection
    mock_db = MagicMock()
    mock_connector.return_value = mock_db

    # Mock the embedding function
    mock_embeddings = MagicMock()
    mock_ollama_embeddings.return_value = mock_embeddings

    # Mock the Chroma vector store
    mock_vector_store = MagicMock()
    mock_chroma.return_value = mock_vector_store

    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.return_value = (
        'You should eat a balanced meal with protein, carbs, and veggies.'
    )
    mock_ollama_llm.return_value = mock_llm

    # Mock the prompt creation
    mock_prompt = MagicMock()
    mock_prompt.input_variables = [
        'context',
        'user',
        'name',
        'height',
        'weight',
        'age',
        'activityLevel',
        'goal',
        'campains',
        'restrictions',
        'user_eaten_food',
        'food',
        'input',
    ]
    mock_chat_prompt.return_value = mock_prompt

    # Mock the doc_chain creation
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = (
        'You should eat a balanced meal with protein, carbs, and veggies.'
    )

    # Mock input data
    mock_data = {
        'content': 'What should I eat today?',
        'username': 'test_user',
    }

    # Call the chat function
    response = chat(mock_data)

    # Assertions
    assert response is not None
    assert (
        response
        == 'You should eat a balanced meal with protein, carbs, and veggies.'
    )
    assert mock_connector.called
    assert mock_ollama_embeddings.called
    assert mock_chroma.called
    assert mock_ollama_llm.called
    assert mock_chat_prompt.called
