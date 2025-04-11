# Development Guide

## Environment Setup
1. Python 3.9+ required
2. Install dependencies: `pip install -r requirements.txt`
3. For development: `pip install -r requirements-dev.txt`

## Project Structure
```
smart_inventory_ai/
├── agents/         # Agent implementations
├── config/         # Configuration files
├── data/           # Sample datasets
├── docs/           # Documentation
├── ml_models/      # Machine learning models
├── tests/          # Unit tests
├── utils/          # Utility functions
└── output/         # Generated reports and outputs
```

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test module
pytest tests/test_llm_interface.py
```

## Code Style
- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Document public interfaces with docstrings
- Keep functions small and focused
