from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from smolagents import Tool


class QdrantQueryTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"
    default_kwargs = {"collection_name": "test", "limit": 5}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = {**self.default_kwargs, **kwargs}
        self.collection_name = (
            self.default_kwargs["collection_name"]
            if kwargs.get("collection_name") is None
            else kwargs["collection_name"]
        )
        self.limit = (
            self.default_kwargs["limit"]
            if kwargs.get("limit") is None
            else kwargs["limit"]
        )

    def forward(self, query: str) -> str:
        embedder = OllamaEmbeddings(model="bge-m3")
        client = QdrantClient()
        points = client.query_points(
            self.collection_name, query=embedder.embed_query(query), limit=self.limit
        ).points
        docs = "Retrieved documents:\n" + "".join(
            [
                f"== Document {str(i)} ==\n" + f"{point.payload['content']}\n"
                for i, point in enumerate(points)
            ]
        )

        return docs


# must have var tools of classes
tools = [QdrantQueryTool]
