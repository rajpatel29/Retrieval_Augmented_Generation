# Retrieval Augmented Generation (RAG) System

A Python-based Retrieval Augmented Generation system that combines vector embeddings with large language models to provide contextually relevant responses.

## Overview

This RAG system uses:
- **Ollama** for local LLM inference and embeddings
- **Qdrant** as a vector database for similarity search
- **mxbai-embed-large** model for generating 1024-dimensional embeddings
- **gemma3:12b-it-qat** model for text generation

## Features

- ✅ Local LLM inference using Ollama
- ✅ Vector embeddings with mxbai-embed-large
- ✅ Vector similarity search with Qdrant
- ✅ Contextual response generation
- ✅ Interactive query interface

## Prerequisites

Before running this system, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running locally
3. **Qdrant** running on localhost:6333

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rajpatel29/Retrieval_Augmented_Generation.git
cd Retrieval_Augmented_Generation
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Required Ollama Models
```bash
# Install embedding model
ollama pull mxbai-embed-large

# Install LLM for text generation
ollama pull gemma3:12b-it-qat
```

### 5. Start Qdrant
```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or using Qdrant binary
# Download from https://qdrant.tech/documentation/guides/installation/
```

## Usage

### Running the RAG System

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the main script:**
   ```bash
   python main.py
   ```

3. **Enter your query when prompted:**
   ```
   Enter a prompt: what do you know about me?
   ```

### Example Queries

The system comes pre-loaded with sample data including:
- "My name is rutvik"
- "I like to play cricket"
- "I like to play football"
- "I am a software engineer"
- "My name is ripal"

Try queries like:
- "Who am I?"
- "What sports do I like?"
- "What is my profession?"
- "Am I ripal or rutvik?"

## How It Works

1. **Data Indexing**: Text data is converted to embeddings using mxbai-embed-large and stored in Qdrant
2. **Query Processing**: User queries are converted to embeddings using the same model
3. **Similarity Search**: Qdrant finds the most similar stored embeddings (top 2 matches)
4. **Context Augmentation**: Retrieved passages are combined with the user query
5. **Response Generation**: gemma3:12b-it-qat generates a contextual response


## Configuration

### Model Configuration
- **Embedding Model**: mxbai-embed-large (1024 dimensions)
- **LLM Model**: gemma3:12b-it-qat
- **Vector Database**: Qdrant with COSINE distance metric
- **Collection Name**: "demo"

### API Endpoints
- **Ollama API**: http://localhost:11434
- **Qdrant API**: http://localhost:6333

## Customization

### Adding New Data
To add your own data, uncomment the indexing section in `main.py` and modify the `dummy_data` list:

```python
dummy_data = [
    "Your custom text here",
    "More custom data",
    # Add more entries...
]
```

### Changing Models
To use different models, update the model names in the API calls:

```python
# For embeddings
json={"model": "your-embedding-model", "input": text}

# For generation
"model": "your-llm-model"
```

## Troubleshooting

### Common Issues

1. **Ollama not running**
   - Ensure Ollama is installed and running: `ollama serve`

2. **Qdrant connection error**
   - Check if Qdrant is running on port 6333
   - Verify with: `curl http://localhost:6333/collections`

3. **Model not found**
   - Install required models: `ollama pull model-name`

4. **Import errors**
   - Activate virtual environment: `source .venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`


