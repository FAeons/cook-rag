import logging

from langchain_openai import ChatOpenAI 
import os
from langchain_core.documents import Document
from typing import List,Generator
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块: 负责生成和集成"""

    def __init__(self, model_name: str = "deepseek-chat",temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块

        Args:
            model_name (str): 生成模型名称
            temperature (float): 生成温度
            max_tokens (int): 生成最大长度
        """

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """设置LLM"""
        logger.info("正在设置LLM...")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            error_msg = "请设置 DEEPSEEK_API_KEY 环境变量"
            logger.error(error_msg)
            raise ValueError(error_msg) 
        
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        if not base_url:
            logger.error("请设置 DEEPSEEK_BASE_URL 环境变量")
            raise ValueError("请设置 DEEPSEEK_BASE_URL 环境变量")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(f"成功加载LLM{self.model_name}")

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成基础答案

        Args:
            question (str): 问题
            context_docs (List[Document]): 上下文文档

        Returns:
            str: 答案
        """

        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")
        
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)

        return response


    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """        
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """重写查询 - 让大模型判断是否要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原始查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询** (直接返回原始查询)
    - 包含具体菜品名称：如"宫保鸡丁怎么做"， "红烧肉的制作方法"
    - 明确的制作询问： 如"蛋炒饭需要什么食材"， "糖醋排骨的制作步骤"
    - 具体的烹饪技巧： 如"如何炒菜不粘锅"， "怎样调制糖醋汁"
2. **模糊的查询** (根据查询内容进行重写)
    - 过于宽泛： 如"做菜"，"有什么好吃的"，"推荐个菜"
    - 缺乏具体信息： 如"川菜"，"素菜"，"简单的"
    - 口语化表达：如"想吃什么"，"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 保持推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做" （保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材" （保持原查询）

请输出最终查询 （如果不需要重写就返回原查询）:""",
            input_variables=["query"],
        )
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()           
        )

        response = chain.invoke(query).strip()

        #记录重写结果
        if response.strip() != query:
            logger.info(f"查询重写：{query} → {response}")
        else:
            logger.info(f"查询无需重写：{query}")
        return response 
    
    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')       
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
    例如： 推荐几个素菜;有什么吃川菜;给我3个简单的菜


2. 'detail' - 用户想要获取菜品的详细制作信息（做法、食材、步骤等）
    例如： 宫保鸡丁怎么做;红烧肉的制作步骤;蛋炒饭需要什么食材
                                                  
3. 'general' - 其它一般性问题
    例如： 什么是川菜;制作技巧;营养价值

请只返回分类结果： list, detail或general

用户问题: {query} 

分类结果:""")
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip().lower()

        if response in ["list", "detail", "general"]:
            return response   
        else:
            return "general"

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:                   
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答        
        """

        if not context_docs:
            return "抱歉，没有找到匹配的菜谱。"

        #提取菜品名称
        dish_names = []

        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        #构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" +  "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" +  "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n...还有{len(dish_names)-3}个菜品可供选择。"

    def generate_basic_stream(self, query: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """
        生成基础流式回答 - 适用于一般性查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            基础流式回答        
        """
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self,query: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """
        生成步骤式回答 - 适用于制作步骤类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            步骤式回答        
        """
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """

        if not docs: 
            return "没有找到匹配的菜谱。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"

            #构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n" + "\n".join(context_parts)
    


if __name__ == "__main__":
    """
    用于测试：
    1. 环境变量是否正确
    2. LLM 是否能成功连接
    3. 模型是否能正常返回内容
    """

    logging.basicConfig(level=logging.INFO)

    try:
        print("🚀 开始测试 LLM 连接...")

        gen = GenerationIntegrationModule(
            model_name="deepseek-chat",
            temperature=0.1,
            max_tokens=200
        )

        test_query = "请用一句话介绍宫保鸡丁"

        print("📨 发送测试请求：", test_query)

        response = gen.llm.invoke(test_query)

        print("\n✅ LLM 连接成功！返回结果如下：\n")
        print(response.content)

    except Exception as e:
        print("\n❌ LLM 连接失败！错误信息：")
        print(e)

