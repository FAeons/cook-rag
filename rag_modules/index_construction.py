"""
构建索引模块
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List
from langchain_core.documents import Document
from pathlib import Path
logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """构建索引模块: 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name (str): 向量化模型名称
            index_save_path (str): 向量索引存储路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.vectorstore = None
        self.embeddings = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """设置向量化模型"""

        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True}
        )
        logger.info(f"成功加载向量化模型{self.model_name}")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """构建向量索引

        Args:
            chunks (List[Document]): 文档分块

        Returns:
            FAISS: 向量存储对象
        """
        logger.info("正在构建向量索引...")

        if not chunks:
            raise ValueError("文档分块列表为空，无法构建索引")
        
        self.vectorstore = FAISS.from_documents(
            documents = chunks, 
            embedding = self.embeddings
        )
        logger.info(f"成功构建向量索引,包含{len(chunks)}个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """向索引中添加新的文档分块

        Args:
            new_chunks (List[Document]): 新的文档分块
        """

        if not self.vectorstore: 
            raise ValueError("请先调用build_vector_index方法构建索引")

        logger.info(f"正在添加{len(new_chunks)}文档分块到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info(f"成功添加文档分块到索引中")

    def save_index(self):
        """保存向量索引"""

        if not self.vectorstore: 
            raise ValueError("请先调用build_vector_index方法构建索引")
        
        #确保保存路径存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)   
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"成功保存向量索引到{self.index_save_path}")

    def load_index(self) -> FAISS | None:
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量索引存储对象，如果加载失败则返回None
        """

        if not self.embeddings:
            self.setup_embeddings()
        
        if not Path(self.index_save_path).exists():
            logger.warning(f"索引路径{self.index_save_path}不存在，无法加载索引")
            return None
        
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.index_save_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"成功从{self.index_save_path}加载向量索引")
            return self.vectorstore
        except Exception as e: 
            logger.warning(f"无法从{self.index_save_path}加载向量索引，错误信息为{e}")
            return None
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """向量索引中搜索最相似的文档

        Args:
            query (str): 查询文本
            k (int, optional): 返回最相似的文档数量。默认为5

        Returns:
            List[Document]: 最相似的文档列表
        """

        if not self.vectorstore: 
            raise ValueError("请先调用build_vector_index或load_index方法构建或加载索引")
        
        return self.vectorstore.similarity_search(query, k)
    
if __name__ == "__main__":
    index_construction = IndexConstructionModule()
    index_construction.load_index()
    doc = index_construction.similarity_search("如何做牛柳")
    count = 0
    for i in doc: 
        count += 1
        print( f"{count}" + i.page_content + "\n")
        

