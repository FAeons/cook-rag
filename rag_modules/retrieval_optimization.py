"""
检索优化模块
"""

import logging
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List,Any,Dict
logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块:负责混合检索和过滤
    """
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore (FAISS): 向量存储
            chunks (List[Document]): 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()


    def setup_retrievers(self):
        """设置检索器"""
        logger.info("正在设置检索器...")

        self.vector_retriever: BaseRetriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.bm25_retriever: BaseRetriever = BM25Retriever.from_documents(
            documents=self.chunks,
            k=5
        )

        logger.info("检索器设置完成！")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """混合搜索
        
        Args:
            query (str): 查询
            top_k (int, optional): 返回结果数量. Defaults to 5.
        
        Returns:
            List[Document]: 搜索结果
        """
        #分别获取向量检索和BM25检索的结果
        #此处有兼容问题，前面建立的retriever可能用不到了
        try:
            # 方案1：向量检索→直接调用FAISS底层方法（最稳定，无参数坑）
            vector_docs = self.vectorstore.similarity_search(query, k=5)
            logger.debug("向量检索成功：使用FAISS.similarity_search")
        except Exception as e:
            logger.error(f"向量检索失败：{e}")
            return []  # 检索失败直接返回空列表，避免后续报错

        try:
            # BM25检索→优先用公开方法，失败则用私有方法（补run_manager=None）
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
            logger.debug("BM25检索成功：使用公开方法get_relevant_documents")
        except AttributeError:
            logger.warning("BM25公开方法不可用，改用私有方法")
            # 调用BM25私有方法时补run_manager=None
            bm25_docs = self.bm25_retriever._get_relevant_documents(query, run_manager=None)
        except TypeError as e:
            logger.error(f"BM25私有方法调用失败：{e}")
            return []

        # 检查检索结果是否为空
        if not vector_docs and not bm25_docs:
            logger.warning("向量检索和BM25检索均无结果")
            return []

        #使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        #logger.info(f"使用RRF重排后的结果数量为{len(reranked_docs)}")
        return reranked_docs[:top_k]
    
    def metadata_filter(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索

        Args:
            query (str): 查询
            filters (Dict[str, Any]): 过滤条件
            top_k (int, optional): 返回结果数量. Defaults to 5.

        Returns:
            List[Document]: 搜索结果
        """

        #先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)

        #使用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        logger.info(f"筛选出{len(filtered_docs)}个结果")
        return filtered_docs


    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF重排文档

        Args:
            vector_docs (List[Document]): 向量检索结果
            bm25_docs (List[Document]): BM25检索结果
            k: int : rrf参数，用于平滑排名
        Returns:
            List[Document]: 重排后的结果    
        """    

        doc_scores = {}
        doc_obj = {}

        #计算向量检索结果的rrf分数
        for rank,doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_obj[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"向量检索结果{doc.page_content}的rrf分数为{rrf_score}")

        #计算BM25检索结果的rrf分数
        for rank,doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_obj[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"BM25检索结果{doc.page_content}的rrf分数为{rrf_score}")

        #根据rrf分数对文档进行排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True) #doc_scores.items()把字典变成一个元组列表

        #最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_obj:
                doc = doc_obj[doc_id]
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 -文档: {doc.page_content[:50]}...最终分数为: {final_score:.4f}")

        logger.info(f"使用RRF重排后的结果数量:向量检索{len(vector_docs)}，BM25检索：{len(bm25_docs)},总结果为:{len(reranked_docs)}")

        return reranked_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from data_preparation import DataPreparationModule
    from index_construction import IndexConstructionModule
    data_prep = DataPreparationModule(data_path="data/cook")
    data_prep.load_documents()
    chunks = data_prep.chunk_documents()

    index_construction = IndexConstructionModule()
    vectorstore = index_construction.load_index()
    test = RetrievalOptimizationModule(vectorstore, chunks)
    doc = test.hybrid_search("如何制作牛排")
    for i in doc: 
        print(i.page_content)
