import unittest
from unittest.mock import patch, MagicMock
from app.models.database import main as app_main 
from langchain.schema.document import Document

class TestPipeline(unittest.TestCase):

    @patch('app_main.PyPDFDirectoryLoader')
    def test_load_documents(self, mock_loader):
        """Test load_documents function."""
        mock_documents = [Document(page_content="Sample text", metadata={"source": "file1.pdf", "page": 1})]
        mock_loader.return_value.load.return_value = mock_documents

        documents = app_main.load_documents()
        self.assertEqual(documents, mock_documents)
        mock_loader.assert_called_once_with(app_main.DATA_PATH)

    @patch('app_main.RecursiveCharacterTextSplitter.split_documents')
    def test_split_documents(self, mock_split_documents):
        """Test split_documents function."""
        mock_documents = [Document(page_content="Sample text", metadata={"source": "file1.pdf", "page": 1})]
        mock_chunks = [Document(page_content="Chunk 1", metadata={"source": "file1.pdf", "page": 1})]
        mock_split_documents.return_value = mock_chunks

        chunks = app_main.split_documents(mock_documents)
        self.assertEqual(chunks, mock_chunks)
        mock_split_documents.assert_called_once_with(mock_documents)

    @patch('app_main.OllamaEmbeddings')
    def test_get_embedding_function(self, mock_embeddings):
        """Test get_embedding_function function."""
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        embeddings = app_main.get_embedding_function()
        self.assertEqual(embeddings, mock_instance)
        mock_embeddings.assert_called_once_with(model="nomic-embed-text")

    @patch('app_main.Chroma')
    @patch('app_main.calculate_chunk_ids')
    def test_add_to_chroma(self, mock_calculate_chunk_ids, mock_chroma):
        """Test add_to_chroma function."""
        mock_chunks = [Document(page_content="Chunk 1", metadata={"id": "file1.pdf:1:0"})]
        mock_calculate_chunk_ids.return_value = mock_chunks

        mock_db = MagicMock()
        mock_db.get.return_value = {"ids": set()}
        mock_chroma.return_value = mock_db

        app_main.add_to_chroma(mock_chunks)

        mock_chroma.assert_called_once_with(
            persist_directory=app_main.CHROMA_PATH, embedding_function=app_main.get_embedding_function()
        )
        mock_db.add_documents.assert_called_once_with(mock_chunks, ids=["file1.pdf:1:0"])
        mock_db.persist.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('shutil.rmtree')
    def test_clear_database(self, mock_rmtree, mock_exists):
        """Test clear_database function."""
        app_main.clear_database()
        mock_exists.assert_called_once_with(app_main.CHROMA_PATH)
        mock_rmtree.assert_called_once_with(app_main.CHROMA_PATH)

    @patch('builtins.print')
    @patch('app_main.add_to_chroma')
    @patch('app_main.split_documents')
    @patch('app_main.load_documents')
    def test_main(self, mock_load_documents, mock_split_documents, mock_add_to_chroma, mock_print):
        """Test the main function."""
        mock_documents = [Document(page_content="Sample text", metadata={})]
        mock_chunks = [Document(page_content="Chunk 1", metadata={})]

        mock_load_documents.return_value = mock_documents
        mock_split_documents.return_value = mock_chunks

        app_main.main()

        mock_load_documents.assert_called_once()
        mock_split_documents.assert_called_once_with(mock_documents)
        mock_add_to_chroma.assert_called_once_with(mock_chunks)
        mock_print.assert_called_with(mock_chunks)

if __name__ == '__main__':
    unittest.main()
