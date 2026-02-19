def make_retriever(vec_db, top_k: int):
    
    return vec_db.as_retriever(search_kwargs={"k": int(top_k)})
