import re
from pathlib import Path
from pypdf import PdfReader

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_documents(folder_path: str):
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        return []

    for file in folder.iterdir():
        content = ""

        if file.suffix.lower() == ".txt":
            content = file.read_text(encoding="utf-8", errors="ignore")

        elif file.suffix.lower() == ".pdf":
            with open(file, "rb") as f:
                reader = PdfReader(f)
                content = " ".join(page.extract_text() or "" for page in reader.pages)

        content = clean_text(content)

        if content:
            docs.append({
                "id": file.stem,
                "text": content
            })

    return docs

