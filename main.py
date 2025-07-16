import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

if not client.collection_exists(collection_name="articles"):
    client.create_collection(
        collection_name="articles",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

# Sample texts to process
dummy_data = [
    "My name is rutvik",
    "I like to play cricket",
    "I like to play football",
    "I am a software engineer",
    "My name is ripal",
]

def main():
    for i, text in enumerate(dummy_data):
        # Get embedding from MxBai
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large", "input": text},
        )
        data = response.json()
        embeddings = data["embeddings"][0]

        client.upsert(
            collection_name="articles",
            points=[PointStruct(id=i, vector=embeddings, payload={"text": text})],
        )

    prompt = input("Enter a prompt: ")
    # it is needed as per embedding LLM's instruction
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": adjusted_prompt},
    )
    data = response.json()
    embeddings = data["embeddings"][0]

    results = client.query_points(
        collection_name="articles",
        query=embeddings,
        with_payload=True,
        limit=2
    )

    for result in results:
        print(result)

if __name__ == "__main__":
    main()