import pytest
from config.settings import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    """Fixture for database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def mock_llm(monkeypatch):
    """Fixture for mocking LLM responses"""
    def mock_query(*args, **kwargs):
        return "Mocked LLM response"
    
    monkeypatch.setattr(
        "agents.interfaces.llm_interface.query_ollama",
        mock_query
    )
