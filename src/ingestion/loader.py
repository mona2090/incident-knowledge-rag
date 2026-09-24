import json
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import PROJECT_ROOT, RAW_DIR


class SourceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    service: str = Field(min_length=1)

    environment: Literal[
        "production", "staging", "development"
    ]

    technology: str = Field(min_length=1)
    document_type: Literal["incident", "runbook"]
    approval_status: Literal["approved", "draft"]
    content: str = Field(min_length=1)


def load_documents() -> list[Document]:
    documents = []
    seen_ids = set()

    for path in sorted(RAW_DIR.rglob("*.json")):
        records = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(records, list):
            raise ValueError(
                f"{path.name} must contain a JSON array"
            )

        for raw_record in records:
            record = SourceRecord.model_validate(raw_record)

            if record.source_id in seen_ids:
                raise ValueError(
                    f"Duplicate source_id: {record.source_id}"
                )

            seen_ids.add(record.source_id)

            metadata = record.model_dump(
                exclude={"content"}
            )
            metadata["source_path"] = (
                path.relative_to(PROJECT_ROOT).as_posix()
            )

            documents.append(
                Document(
                    page_content=(
                        f"{record.title}\n\n{record.content}"
                    ),
                    metadata=metadata,
                )
            )

    if not documents:
        raise ValueError(
            "No source documents found in data/raw"
        )

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    for doc in docs:
        print(
            doc.metadata["source_id"],
            doc.metadata["title"],
        )