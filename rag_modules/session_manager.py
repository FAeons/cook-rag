"""
会话管理模块 - 支持多轮对话

=== 这个模块是做什么的？ ===
管理用户的对话历史，让AI能够理解上下文、支持追问

=== 多轮对话的重要性 ===
单轮对话：
  用户：红烧肉怎么做？ → AI回答
  用户：用什么调料？ → AI不知道"用什么调料"是问红烧肉还是别的

多轮对话：
  用户：红烧肉怎么做？ → AI回答
  用户：用什么调料？ → AI知道是在问红烧肉的调料

=== 使用示例 ===
session = SessionManager()
session_id = session.create_session("user123")
session.add_message(session_id, "user", "红烧肉怎么做")
session.add_message(session_id, "assistant", "红烧肉的做法是...")
context = session.get_context(session_id)  # 获取对话上下文
"""


import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """消息对象"""
    role: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """会话对象"""
    session_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0


class SessionManager:
    """
    会话管理器 - 管理多轮对话

    === 核心功能 ===
    1. 创建/获取会话
    2. 添加消息到会话
    3. 获取对话上下文（用于构建prompt）
    4. 会话过期清理

    === 线程安全 ===
    使用锁保证多线程环境下的安全
    """

    def __init__(
        self,
        max_sessions: int = 1000, # 最大会话数
        max_history: int = 10,   # 每个会话保留的最大消息轮数
        session_ttl: int = 3600,    # 会话过期时间（秒）
        context_window: int = 5   # 构建上下文时使用的最近轮数
    ):
        """
        初始化会话管理器

        Args:
            max_sessions: 最大同时存在的会话数（LRU淘汰）
            max_history: 每个会话保留的最大消息轮数
            session_ttl: 会话过期时间（秒），超时自动清理
            context_window: 构建上下文时使用最近几轮对话
        """

        self.max_sessions = max_sessions
        self.max_history = max_history
        self.session_ttl = session_ttl
        self.context_window = context_window

        # 使用OrderedDict实现LRU缓存
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.RLock()

        logger.info(
            f"会话管理器初始化完成: max_sessions={max_sessions}, max_history={max_history}")

    def create_session(self, user_id: str = "anonymous") -> str:
        """
        创建新会话

        Args:
            user_id (str, optional): 用户ID. Defaults to "anonymous".
        Returns:
            str: 会话ID
        """
        with self._lock:
            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                user_id=user_id
            )

            # 添加到会话字典
            self._sessions[session_id] = session

            logger.info(f"创建新会话: session_id={session_id}, user_id={user_id}")

            while len(self._sessions) > self.max_sessions:
                oldest_id = next(iter(self._sessions))
                del self._sessions[oldest_id]
                logger.debug(f"LRU淘汰会话: {oldest_id}")

            return session_id
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取指定会话

        Args:
            session_id (str): 会话ID

        Returns:
            Optional[Session]: 如果不存在或过期返回None
        """

        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                return None
            
            # 检查是否过期
            if time.time() - session.updated_at > self.session_ttl:
                logger.info(f"会话已过期: {session_id}")
                del self._sessions[session_id]
                return None
            
            # 移动到OrderedDict末尾（更新LRU顺序）
            self._sessions.move_to_end(session_id)
            
            return session
    
    def get_or_create_session(self, session_id: str = None, user_id: str = "anonymous") -> str:
        """
        获取会话，如果不存在则创建

        Args: 
            session_id (str): 会话ID
            user_id (str, optional): 用户ID. Defaults to "anonymous".

        Returns:
            str: 会话ID
        """
        with self._lock:
            if session_id:
                session = self.get_session(session_id)
                if session:
                    return session.session_id
            return self.create_session(user_id)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        添加消息到会话

        Args:
            session_id (str): 会话ID
            role (str): 消息角色
            content (str): 消息内容
            metadata (Dict[str, Any], optional): 消息元数据. Defaults to None.

        Returns:
            bool: 是否添加成功
        """

        with self._lock:
            session = self.get_session(session_id)
            if session is None:
                logger.warning(f"会话不存在: {session_id}")
                return False
            
            message = Message(
                role=role,
                content=content,
                timestamp=time.time(),
                metadata=metadata or {}
            )


            session.messages.append(message)
            session.updated_at = time.time()


            #   如果消息数超过限制，删除最早的消息
            while len(session.messages) > self.max_history * 2:
                session.messages.pop(0)
                session.messages.pop(0)   

            logger.debug(f"添加消息到会话 {session_id}: [{role}] {content[:50]}...")
            return True       

    def get_context(self,session_id: str, include_current: bool = False) -> str:
        """
        获取会话上下文(用于构建prompt)

        Args:
            session_id (str): 会话ID
            include_current (bool): 是否包含最后一条用户信息
        Returns:
            str: 会话上下文
        """

        with self._lock:
            session = self.get_session(session_id)

            if session is None or session.is_empty:
                logger.warning(f"会话不存在或没有历史聊天记录: {session_id}")
                return ""
            
            # 获取最近的对话轮次
            messages = session.messages
            if not include_current and messages and messages[-1].role == 'user':
                messages = messages[:-1]

            # 只取最近的context_window轮对话
            recent_messages = messages[-(self.context_window * 2):]

            if not recent_messages:
                return ""


            #格式化上下文
            context_parts = []
            for msg in recent_messages:
                role_name = '用户' if msg.role == 'user' else '助手'
                context_parts.append(f"{role_name}: {msg.content}") 


            return "\n".join(context_parts)


    def get_messages(self, session_id: str, limit: int = None) -> List[Dict [str, Any]]:
        """
        获取会话消息

        Args:
            session_id (str): 会话ID
            limit (int, optional): 返回消息数量限制. Defaults to None.

        Returns:
            消息列表，格式：[{"role": "user", "content": "..."}, ...]
        """
        with self._lock:
            session = self.get_session(session_id)

            if session is None:
                return []
            
            messages = session.messages
            if limit:
                messages = messages[-limit:]

            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

        
    def clear_session(self, session_id: str) -> bool:
        """
        清空会话

        Args:
            session_id (str): 会话ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            session = self.get_session(session_id)

            if session is None:
                return False
            
            session.messages = []
            session.updated_at = time.time()
            logger.info(f"会话已清空: {session_id}")
            return True
        
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id (str): 会话ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"会话已删除: {session_id}")
                return True
            logger.warning(f"会话不存在: {session_id}")
            return False
        

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息

        Args:
            session_id (str): 会话ID

        Returns:
            Dict[str, Any]: 会话信息
        """
        with self._lock:
            session = self.get_session(session_id)

            if session is None:
                return None
            
            return {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'created_at': session.created_at,
                'updated_at': session.updated_at,
                'messages_count': session.message_count,
            }
        
    def cleanup_expired(self) -> int:
        """
        清理过期会话

        Returns:
            int: 清理的会话数量
        """

        with self._lock:
            now = time.time()
            expired = []

            for session_id, session in self._sessions.items():
                if now - session.updated_at > self.session_ttl:
                    expired.append(session_id)

            for session_id in expired:
                self.delete_session(session_id)

            if expired:
                logger.info(f"清理过期会话: {len(expired)}个会话")

            return len(expired)
        
    @property
    def active_session_count(self) -> int:
        """
        获取当前活跃会话数量

        Returns:
            int: 活跃会话数量
        """
        return len(self._sessions)
    

# 全局会话管理器实例
_session_manager: Optional[SessionManager] = None


def get_session_manager(
    max_sessions: int = 1000,
    max_history: int = 10,
    session_ttl: int = 3600,
    context_window: int = 5
) -> SessionManager:
    """
    获取全局会话管理器实例（单例模式）
    修复：支持自定义初始化参数，提升灵活性

    Args:
        max_sessions: 最大同时存在的会话数
        max_history: 每个会话保留的最大消息轮数
        session_ttl: 会话过期时间（秒）
        context_window: 构建上下文时使用的最近轮数

    Returns:
        SessionManager: 全局单例会话管理器
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(
            max_sessions=max_sessions,
            max_history=max_history,
            session_ttl=session_ttl,
            context_window=context_window
        )
    return _session_manager
         
            
# 测试会话管理器
if __name__ == "__main__":
    # 获取管理器（自定义配置）
    sm = get_session_manager(max_history=5, session_ttl=3600)
    
    # 创建会话
    session_id = sm.create_session("test_user")
    
    # 添加消息
    sm.add_message(session_id, "user", "红烧肉怎么做？")
    sm.add_message(session_id, "assistant", "红烧肉的做法分为选肉、焯水、炒糖色...")
    sm.add_message(session_id, "user", "用什么调料？")
    
    # 获取上下文
    context = sm.get_context(session_id)
    print("对话上下文：")
    print(context)
    
    # 获取消息列表
    messages = sm.get_messages(session_id)
    print("\n消息列表：")
    for msg in messages:
        print(f"[{msg['role']}] {msg['content']}")
    
    # 清理过期会话
    sm.cleanup_expired()
    print(f"\n活跃会话数：{sm.active_session_count}")   

        
