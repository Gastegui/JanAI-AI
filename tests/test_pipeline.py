import unittest
from unittest.mock import patch, MagicMock
from app.models.database import load_documents, split_documents, get_embedding_function, add_to_chroma, clear_database, main
from langchain.schema.document import Document

CHROMA_PATH = "chroma"
DATA_PATH = "data"  

class TestPipeline(unittest.TestCase):

    @patch('app.models.database.PyPDFDirectoryLoader')
    def test_load_documents(self, mock_loader):
        """Test load_documents function."""
        mock_documents = [Document(page_content="Sample text", metadata={"source": "file1.pdf", "page": 1})]
        mock_loader.return_value.load.return_value = mock_documents

        documents = load_documents()
        self.assertEqual(documents, mock_documents)
        mock_loader.assert_called_once_with(DATA_PATH)

    @patch('app.models.database.RecursiveCharacterTextSplitter.split_documents')
    def test_split_documents(self, mock_split_documents):
        """Test split_documents function."""
        mock_documents = [Document(page_content="Sample text", metadata={"source": "file1.pdf", "page": 1})]
        mock_chunks = [Document(page_content="Chunk 1", metadata={"source": "file1.pdf", "page": 1})]
        mock_split_documents.return_value = mock_chunks

        chunks = split_documents(mock_documents)
        self.assertEqual(chunks, mock_chunks)
        mock_split_documents.assert_called_once_with(mock_documents)

    @patch('app.models.database.OllamaEmbeddings')
    def test_get_embedding_function(self, mock_embeddings):
        """Test get_embedding_function function."""
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        embeddings = get_embedding_function()
        self.assertEqual(embeddings, mock_instance)
        mock_embeddings.assert_called_once_with(model="nomic-embed-text")

    @patch('app.models.database.get_embedding_function') 
    @patch('app.models.database.Chroma')
    @patch('app.models.database.calculate_chunk_ids')
    def test_add_to_chroma(self, mock_calculate_chunk_ids, mock_chroma, mock_get_embedding_function):
        """Test add_to_chroma function."""
        mock_chunks = [Document(page_content="Chunk 1", metadata={"id": "file1.pdf:1:0"})]
        mock_calculate_chunk_ids.return_value = mock_chunks

        mock_embedding_function = MagicMock()
        mock_get_embedding_function.return_value = mock_embedding_function

        mock_db = MagicMock()
        mock_db.get.return_value = {"ids": set()}
        mock_chroma.return_value = mock_db

        add_to_chroma(mock_chunks)

        mock_chroma.assert_called_once_with(
            persist_directory=CHROMA_PATH, embedding_function=mock_embedding_function
        )
        mock_db.add_documents.assert_called_once_with(mock_chunks, ids=["file1.pdf:1:0"])
        mock_db.persist.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('shutil.rmtree')
    def test_clear_database(self, mock_rmtree, mock_exists):
        """Test clear_database function."""
        clear_database()
        mock_exists.assert_called_once_with(CHROMA_PATH)
        mock_rmtree.assert_called_once_with(CHROMA_PATH)

    @patch('builtins.print')
    @patch('app.models.database.add_to_chroma')
    @patch('app.models.database.split_documents')
    @patch('app.models.database.load_documents')
    def test_main(self, mock_load_documents, mock_split_documents, mock_add_to_chroma, mock_print):
        """Test the main function."""
        mock_documents = [Document(page_content="Sample text", metadata={})]
        mock_chunks = [Document(page_content="Chunk 1", metadata={})]

        mock_load_documents.return_value = mock_documents
        mock_split_documents.return_value = mock_chunks

        main()

        mock_load_documents.assert_called_once()
        mock_split_documents.assert_called_once_with(mock_documents)
        mock_add_to_chroma.assert_called_once_with(mock_chunks)
        mock_print.assert_called_with(mock_chunks)

if __name__ == '__main__':
    unittest.main()
