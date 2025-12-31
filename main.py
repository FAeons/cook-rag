"""
RAG主程序
"""


import os
import sys
import logging
from pathlib import Path
from typing import List,Generator

# 添加模块路径到系统路径

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import(
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    get_cache_manager,
    get_session_manager
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """
    RecipeRAG系统
    """
    def __init__(self, config: RAGConfig = None):
        """
        初始化RecipeRAG系统
        
        Args:
            config (RAGConfig): 配置实例,默认使用DEFAULT_CONFIG
        """

        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        self.session_manager = get_session_manager()
        self.cache_manager = get_cache_manager()

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise ValueError(f"数据路径{self.config.data_path}不存在")
        
        # 检查API密钥   
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置LLM API密钥")
        
        # 检查Base URL
        if not os.getenv("DEEPSEEK_BASE_URL"):
            raise ValueError("请设置OpenAI API Base URL")
        

    def initialize_system(self):
        """
        初始化系统
        """
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化向量检索优化模块·
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

    def build_knowledge_base(self):
        """
        构建知识库
        """
        print("\n🧠 正在构建知识库...")
        

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2.加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3.进行文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4.构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5.保存向量索引
            print("保存向量索引...")
            self.index_module.save_index()
        
        # 6.初始化向量检索模块
        print("初始化向量检索优化模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore=vectorstore, chunks=chunks)


        # 7. 显示统计信息
        stats = self.data_module.get_statistics()  
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    # 新增查询补全函数，防止Router & Rewrite 被污染
    def compose_query(self, session_id: str, question: str) -> str:
        context = self.session_manager.get_context(session_id)

        if not context:
            return question

        prompt = f"""
    根据下面的对话上下文，补全用户的当前问题，使它成为一个完整、可独立理解的查询。

    对话上下文:
    {context}

    用户当前问题:
    {question}

    补全后的完整问题:
    """
        return self.generation_module.llm.invoke(prompt).content.strip()


    # 抽离流式输出逻辑为独立方法
    def _ask_question_stream(self, full_question: str, question: str, session_id: str, relevant_docs: list, route_type: str) -> Generator[str, None, None]:
        """内部流式回答生成方法"""
        if not relevant_docs:
            yield "未找到该菜品的详细信息，请尝试其他关键词"
            return
        
        try:
            if route_type == "detail":
                buffer = []
                for chunk in self.generation_module.generate_step_by_step_answer_stream(full_question, relevant_docs):
                    buffer.append(chunk)
                    yield chunk
                answer = "".join(buffer)
            else:
                buffer = []
                for chunk in self.generation_module.generate_basic_stream(full_question, relevant_docs):
                    buffer.append(chunk)
                    yield chunk
                answer = "".join(buffer)
            
            # 保存会话和缓存
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
        except Exception as e:
            print(f"⚠️ 流式生成回答失败: {e}")
            yield "生成回答失败，请重试"


    def ask_question(self, question: str, session_id: str, stream: bool = False) -> str | Generator[str, None, None]:
        """
        问答主方法
            
        Args:
            question (str): 用户问题
            session_id (str): 会话ID
            stream (bool): 是否流式返回答案
        
        Returns:
            str | Generator: 非流式返回字符串，流式返回生成器
        
        Raises:
            ValueError: 如果没有构建知识库
        """
        # 1. 查缓存
        cached = self.cache_manager.get(session_id, question)
        if cached is not None:
            print(f"⚡ 命中缓存, 内容前50字: {cached[:50]}")
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", cached)
            
            return cached

        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")
        full_question = self.compose_query(session_id, question)

        # 1.查询路由
        route_type = self.generation_module.query_router(full_question)
        print(f"🎯 查询类型: {route_type}")

        # 2.智能查询重写(根据路由类型)
        if route_type == "list":
            rewritten_question = full_question
            print(f"📝 列表查询保持原样: {rewritten_question}")
        else:
            print("🤖 智能分析查询...")
            rewritten_question = self.generation_module.query_rewrite(full_question)
            print(f"📝 智能重写后的查询: {rewritten_question}")
        
        # 3.检索相关子块(自动应用元数据过滤)    
        print("🔍 检索相关子块...")
        filters = self._extract_filters_from_query(full_question)
        if filters:
            print(f"📝 应用的过滤器: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filter(rewritten_question, filters, top_k=self.config.top_k)
        else:   
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_question, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                content_preview = chunk.page_content[:50].replace("\n", " ").strip()
                if content_preview.startswith("#"):
                    title_end = content_preview.find('\n') if '\n' in chunk.page_content[:100] else len(content_preview)
                    section_title = content_preview[:title_end].strip('#').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")
            print(f"📚 检索到的子块: {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4.检查是否找到相关内容
        if not relevant_chunks:
            print("⚠️ 没有找到任何相关文档块，请重新提问。")
            answer = "没有找到任何相关文档块，请尝试其他菜品名称或关键词"
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
            return answer
        
        # ========== 关键修改1：提前初始化relevant_docs，确保作用域覆盖所有分支 ==========
        relevant_docs = []
        try:
            # 获取父文档（核心：这行代码必须在所有使用relevant_docs的分支之前执行）
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        except Exception as e:
            print(f"⚠️ 获取父文档失败: {e}")
            answer = "获取菜谱详情失败，请重试"
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
            return answer

        # 显示找到的文档名称
        doc_names = [doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs] if relevant_docs else []
        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")
        else:
            print(f"对应 {len(relevant_docs)} 个完整文档")

        # 5.列表查询分支
        if route_type == "list":
            print("📝 返回菜品名称列表...")
            answer = self.generation_module.generate_list_answer(full_question, relevant_docs)
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
            return answer

        # 6.详细/一般查询分支
        print("✍️ 生成详细回答...")
        # ========== 关键修改2：增加relevant_docs非空检查，防止空列表导致生成失败 ==========
        if not relevant_docs:
            answer = "未找到该菜品的详细信息，请尝试其他关键词"
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
            return answer

        if stream:
            # 返回流式生成器
            return self._ask_question_stream(full_question, question, session_id, relevant_docs, route_type)
        else:
            # 非流式：直接生成并返回字符串
            try:
                if route_type == "detail":
                    answer = self.generation_module.generate_step_by_step_answer(full_question, relevant_docs)
                else:
                    answer = self.generation_module.generate_basic_answer(full_question, relevant_docs)
            except Exception as e:
                print(f"⚠️ 生成回答失败: {e}")
                answer = "生成菜谱回答失败，请重试"
            
            # 保存会话和缓存
            self.session_manager.add_message(session_id, "user", question)
            self.session_manager.add_message(session_id, "assistant", answer)
            self.cache_manager.set(session_id, question, answer, metadata={"route": route_type})
            return answer
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}

        # 分类关键词
        category_keywords = DataPreparationModule.get_category_labels() 
        for category in category_keywords:
            if category in query: 
                filters["category"] = category
                break   

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_difficulty_labels()
        for difficulty in difficulty_keywords:
            if difficulty in query:
                filters["difficulty"] = difficulty
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先初始化检索模块")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}

        docs = self.retrieval_module.metadata_filter(search_query, filters, top_k=10)


        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names
    def get_ingredients(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]) :
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"请提供 {dish_name} 的食材信息", docs)  

        return answer

    def run_interactive(self):
        """
        交互式问答
        """
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")  

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")


        # 创建一个会话，生成session_id(LRU 是对 所有 session 混在一起 做的 -> TODO:每个 session 一个 LRU 缓存桶)
        session_id = self.session_manager.create_session("cli_user")

        while True:
            try:
                user_input = input("\n您的问题：").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    print("byebye~")
                    break   
                    
                # 询问是否使用流式输出
                stream_chice = input("是否使用流式输出?(y/n, 默认y): ").strip().lower()
                user_stream = stream_chice != 'n'

                print("\n回答:")
                if user_stream:
                    # 流式输出：迭代生成器
                    response_generator = self.ask_question(user_input, session_id, stream=True)
                    for chunk in response_generator:
                        print(chunk, end='', flush=True)
                    print("\n")
                else:
                    # 非流式输出：直接获取字符串
                    answer = self.ask_question(user_input, session_id, stream=False)
                    print(answer)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"发生错误: {e}")

        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        # 创建一个RAG系统实例
        rag_system = RecipeRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"发生错误: {e}")
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()

        
