import pytest
from agents.interfaces.llm_interface import query_ollama

def test_query_ollama_mocked(mock_llm):
    """Test LLM interface with mocked response"""
    result = query_ollama("test prompt")
    assert result == "Mocked LLM response"

def test_query_ollama_error(monkeypatch):
    """Test LLM interface error handling"""
    def mock_error(*args, **kwargs):
        raise Exception("Test error")
    
    monkeypatch.setattr(
        "agents.interfaces.llm_interface.query_ollama",
        mock_error
    )
    result = query_ollama("test prompt")
    assert "not responding" in result
