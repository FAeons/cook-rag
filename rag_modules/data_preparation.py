
from typing import List, Dict, Any
from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
import hashlib
import uuid
import json
logger = logging.getLogger(__name__)

class DataPreparationModule:
    CATAGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }

    CATAGORY_LABELS = list(set(CATAGORY_MAPPING.values()))

    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.child_to_parent: Dict[str, str] = {}
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据

        Returns:
            List[Document]: 加载的文档列表
        """
        logger.info(f"正在从{self.data_path}加载文档")

        documents = []

        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()

                    try: 
                        data_root = Path(self.data_path).resolve()
                        relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                    except Exception as e:
                        relative_path = Path(md_file).as_posix() #兜底 
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest() #把字符串路径转换成字节，用md5算法算一个hash值，再转成16进制字符串的形式

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(md_file),
                            "parent_id": parent_id,
                            "doc_type": "parent"
                        }
                    )
                    documents.append(doc)

            except Exception as e: 
                logger.warning(f"无法处理文件{md_file}，错误信息为{e}")

        #增强元数据
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载了{len(documents)}个文档")
        return documents
    
    def _enhance_metadata(self, doc: Document):
        """
        增强元数据

        Args:
            doc (Document): 要增强元数据的文档对象
        """

        file_path = Path(doc.metadata.get("source", ''))
        file_parts = file_path.parts # 路径切分后的元组

        doc.metadata['category'] = "其他"  
        for part in file_parts:
            if part in self.CATAGORY_MAPPING:
                doc.metadata['category'] =  self.CATAGORY_MAPPING[part]
                break

        doc.metadata['dish_name'] = file_path.stem

        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_category_labels(cls) -> List[str]:
        """
        获取类别标签列表

        Returns:
            List[str]: 类别标签列表
        """
        return cls.CATAGORY_LABELS
    
    @classmethod
    def get_difficulty_labels(cls) -> List[str]:
        """
        获取难度标签列表

        Returns:
            List[str]: 难度标签列表
        """
        return cls.DIFFICULTY_LABELS
    
    def chunk_documents(self) -> List[Document]:
        """
        对Markdown文档进行分块处理

        Returns:
            List[Document]: 分块后的文档列表
        """
        
        logger.info("正在对文档进行分块处理")

        if not self.documents: 
            raise ValueError("请先调用load_documents方法加载文档")
        
        chunks = self._markdown_header_split()

        #为每个chunk添加元数据
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                #如果没有chunk_id,则说明是未分割的原始文档，则生成一个
                chunk.metadata["chunk_id"] = str(uuid.uuid4())    #每个父document都有parent_id，每个chunk都有chunk_id和parent_id。包括未被切分的文档
            chunk.metadata["batch_index"] = i #全局唯一的索引
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"成功对{len(self.documents)}个文档进行分块处理，生成了{len(chunks)}个块")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            List[Document]: 分块后的文档列表
        """ 

        header_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        #下层标题所在内容块，会在 metadata 中携带它路径上的所有上层标题”
        markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=header_to_split_on,
            strip_headers=False, #为False，则保留标题且；page.content也会保留标题文本
        )

        all_chunks = []

        for doc in self.documents:
            try:
                #检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))
                if not has_headers:
                    logger.warning(f"文档{doc.metadata.get('dish_name', '未知')}没有发现Markdown标题")
                    logger.warning(f"预览内容如下：\n{content_preview}")
                
                #对每个文档进行markdown分割
                md_chunks = markdown_header_splitter.split_text(doc.page_content)

                logger.debug(f"文档{doc.metadata.get('dish_name', '未知')}分割成了{len(md_chunks)}个块")

                #如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1: 
                    logger.warning(f"文档{doc.metadata.get('dish_name', '未知')}没有成功进行结构化分割，请检查文档结构")


                #为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "doc_type": "child",
                        "parent_id": parent_id,
                        "chunk_id": child_id,
                        "chunk_index": i,
                    })
                    self.child_to_parent[child_id] = parent_id

                all_chunks.extend(md_chunks) #一个一个加进去

            except Exception as e:
                logger.warning(f"无法分割Markdown文件{doc.metadata.get('dish_name', '未知')}，错误信息为{e}")
                all_chunks.append(doc)
        
        logger.info(f"成功分割出{len(all_chunks)}个文档块")
        return all_chunks
    
    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        根据类别过滤文档

        Args:
            category (str): 要过滤的类别标签

        Returns:
            List[Document]: 过滤后的文档列表
        """
        if not self.documents:
            raise ValueError("请先调用load_documents方法加载文档")
        
        filtered_docs = [doc for doc in self.documents if doc.metadata.get("category") == category]
        logger.info(f"根据类别'{category}'过滤后，得到{len(filtered_docs)}个文档")
        return filtered_docs

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        根据难度过滤文档

        Args:
            difficulty (str): 要过滤的难度标签

        Returns:
            List[Document]: 过滤后的文档列表
        """
        if not self.documents:
            raise ValueError("请先调用load_documents方法加载文档")
        filtered_docs = [doc for doc in self.documents if doc.metadata.get("difficulty") == difficulty]
        logger.info(f"根据难度'{difficulty}'过滤后，得到{len(filtered_docs)}个文档")
        return filtered_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        if not self.documents:
            raise ValueError("请先调用load_documents方法加载文档")
        

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计类别
            category = doc.metadata.get("category","未知")
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get("difficulty","未知")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1


        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
            "avg_chunk_size": sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    

    def export_metadata(self, output_file: str) -> None:
        """
        导出文档元数据

        Args:
            output_file (str): 导出的文件路径
        """
        
        if not self.documents:
            raise ValueError("请先调用load_documents方法加载文档")
        
        metadata_list = []

        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get("source", ""),
                'dish_name': doc.metadata.get("dish_name", ""),
                'category': doc.metadata.get("category", ""),
                'difficulty': doc.metadata.get("difficulty", ""),
                'content_length': len(doc.page_content),
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"成功导出文档元数据到{output_file}")

    

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        获取父文档列表(智能去重)
        Args:
            child_chunks (List[Document]): 检索到的子文档列表
        Returns:
            List[Document]: 对应的父文档列表(去重，按相关性排序)
        """
        
        #统计每个父文档被匹配的次数(相关性指标)
        parent_relevance = {}
        parent_doc_map = {}


        #收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                #增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                if parent_id not in parent_doc_map:
                    for doc in self.documents:
                        #缓存父文档，避免重复查找
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_doc_map[parent_id] = doc
                            break
                            
        #根据相关性分数排序父文档ID
        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True
        )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_doc_map:  #防御性编程，确保存在
                parent_docs.append(parent_doc_map[parent_id])

        
        #收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name", "未知")
            current_parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(current_parent_id, 0)            
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从{len(child_chunks) }个子文档中，找到{len(parent_docs)}个父文档：{parent_info})")    
        return parent_docs
    
if __name__ == "__main__":
    data_prep = DataPreparationModule(data_path="data/cook")
    data_prep.load_documents()
    data_prep.chunk_documents()
    stats = data_prep.get_statistics()
    print("统计信息：")
    for key, value in stats.items():
        print(f"{key}: {value}\n")
    data_prep.export_metadata(output_file="document_metadata.json")