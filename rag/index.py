from langchain_community.vectorstores import FAISS


def build_faiss_index(chunks, embeddings):

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    vec_db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)
    return vec_db
