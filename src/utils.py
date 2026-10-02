import re


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def clean_extracted_text(text: str) -> str:
    # Replace multiple whitespace with single spaces
    cleaned = re.sub(r'\s+', ' ', text)
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
    # Strip leading/trailing whitespace
    return cleaned.strip()
