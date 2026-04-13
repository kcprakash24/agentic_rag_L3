import re


# Keyword patterns → collection name
# Add more rules as you ingest different types of papers
COLLECTION_RULES = [
    (r"transformer|attention|bert|gpt|llm|language.model", "nlp"),
    (r"vision|image|cnn|convolution|resnet|vit", "computer_vision"),
    (r"reinforcement|reward|policy|agent|rl", "reinforcement_learning"),
    (r"diffusion|stable.diffusion|generative|gan|vae", "generative_models"),
    (r"graph|node|edge|gnn|network", "graph_learning"),
]


def route_document(file_name: str, markdown_text: str = "") -> str:
    """
    Assign a document to a collection based on filename and content.

    Checks filename first (fast), then scans first 2000 chars of
    content for keyword matches. Falls back to 'general'.

    Args:
        file_name: PDF filename
        markdown_text: Parsed markdown content (optional)

    Returns:
        Collection name string
    """
    # Combine filename + start of content for matching
    text_to_check = (file_name + " " + markdown_text[:2000]).lower()

    for pattern, collection in COLLECTION_RULES:
        if re.search(pattern, text_to_check):
            print(f"  Routed '{file_name}' → collection: '{collection}'")
            return collection

    print(f"  Routed '{file_name}' → collection: 'general' (no match)")
    return "general"


def get_all_collections() -> list[str]:
    """Return all known collection names."""
    collections = list({c for _, c in COLLECTION_RULES})
    collections.append("general")
    return sorted(collections)