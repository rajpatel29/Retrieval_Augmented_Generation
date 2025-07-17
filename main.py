# Import necessary libraries
import requests  # For sending HTTP requests to local LLM and embedding services
from qdrant_client import QdrantClient  # Main client to interact with Qdrant vector database
from qdrant_client.models import Distance, VectorParams, PointStruct  # Qdrant model classes

# Initialize Qdrant client to connect to local Qdrant server
client = QdrantClient("localhost", port=6333)

# Check if the collection "demo" exists, if not create it
if not client.collection_exists(collection_name="demo"):
    client.create_collection(
        collection_name="demo",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  # Vector size must match the embedding output
    )

# List of sample texts to embed and store in the Qdrant vector database
dummy_data = [
    "My name is rutvik",
    "I like to play cricket",
    "I like to play football",
    "I am a software engineer",
    "My name is ripal",
]

# Function to generate a response from a local LLM using the given prompt
def generate_response(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",  # Local inference API endpoint
        json={
            "model": "gemma3:12b-it-qat",  # Model name to use for generation
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 10000  # Context length
            }
        }
    )
    response_data = response.json()
    print("Debug - Response structure:", response_data.keys())  # Optional: print keys for debugging
    return response_data.get("response", "No response generated")  # Return generated text

# Main function to execute embedding and querying workflow
def main():
    # Step to index dummy_data in Qdrant is commented out after initial run
    for i, text in enumerate(dummy_data):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large", "input": text},
        )
        data = response.json()
        embeddings = data["embeddings"][0]

        # Store embeddings in Qdrant collection
        client.upsert(
            collection_name="demo",
            points=[PointStruct(id=i, vector=embeddings, payload={"text": text})],
        )

    # Take user input prompt
    prompt = input("Enter a prompt: ")

    # Adjust prompt according to embedding model’s expected format
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"
    
    # Get embedding for the adjusted prompt
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": adjusted_prompt},
    )
    data = response.json()
    embeddings = data["embeddings"][0]

    # Query Qdrant for the most relevant stored texts (top 2 matches)
    results = client.query_points(
        collection_name="demo",
        query=embeddings,
        with_payload=True,
        limit=2
    )

    # Extract matched text payloads from retrieved points
    relevant_passages = "\n".join(
        [f"- {point.payload['text']}" for point in results.points if point.payload and 'text' in point.payload])

    # Construct a new prompt including both user question and retrieved context
    augmented_prompt = f"""
      The following are relevant passages:
      <retrieved-data>
      {relevant_passages}
      </retrieved-data>

      Here's the original user prompt, answer with help of the retrieved passages:
      <user-prompt>
      {prompt}
      </user-prompt>
    """

    # Generate answer using local LLM with augmented prompt
    response = generate_response(augmented_prompt)
    print(response)

# Run the main function when script is executed
if __name__ == "__main__":
    main()
