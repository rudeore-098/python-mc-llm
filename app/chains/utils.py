def format_docs(docs) -> str:
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )
