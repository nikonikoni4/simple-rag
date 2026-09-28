"""
综合检索
"""
from simple_rag.repository.vec import Schema, VecDB, VecSearchResult
from simple_rag.embedding_api import DoubaoEmbeddingVision
# 注意，当前embed客户端只有DoubaoEmbeddingVision，若后续扩张到多个embed provider 需要
# 做底层集合，做一个共同的clietn

def _desen_retrival(vec_db,q):
    """
    稠密检索
    
    """


class RetrievalClient:
    """
    要考虑这个检索客户端到底要不要依赖注入
    
    """
    def __init__(self,vec_db:VecDB,embedding_client:DoubaoEmbeddingVision):
        self.embedding_client = embedding_client
        self.vec_db = vec_db
        
        pass

    def search(self,query:str):
        pass

    