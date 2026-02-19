from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_docs(docs, chunk_size: int, chunk_overlap: int, separators=None, add_start_index=True):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ".", " ", ""],
        add_start_index=add_start_index,
    )
    return splitter.split_documents(docs)
