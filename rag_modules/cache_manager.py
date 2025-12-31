"""
缓存管理模块 - 提升查询效率

=== 这个模块是做什么的？ ===
缓存热点查询的结果，避免重复计算

=== 为什么需要缓存？ ===
1. RAG检索+生成需要几秒到十几秒
2. 很多用户会问相似的问题
3. 缓存后直接返回，毫秒级响应

=== 缓存策略 ===
1. 精确匹配缓存：完全相同的问题直接命中
2. 语义相似缓存：意思相近的问题也能命中
3. LRU淘汰：最久未使用的缓存优先删除
4. TTL过期：缓存有过期时间，保证数据新鲜度

=== 使用示例 ===
cache = CacheManager()
cache.set("红烧肉怎么做", "红烧肉的做法是...")
answer = cache.get("红烧肉怎么做")  # 命中缓存
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import threading
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    answer: str
    created_at: float
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheManager:
    """
    缓存管理器

    - LRU（最近最少使用）淘汰
    - TTL（过期时间）
    - 命中率统计
    """

    def __init__(
        self,
        max_size: int = 1000,  # 缓存最大条目数
        ttl: int = 3600,          # 缓存过期时间（秒），默认1小时
        enable_semantic: bool = False,  # 预留扩展,语义相似缓存
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.enable_semantic = enable_semantic  # 语义缓存开关（预留，需结合嵌入模型实现）

        # OrderedDict实现LRU：命中的条目移到末尾，淘汰时删除头部
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self._stats = {
            "hits": 0, # 缓存命中次数
            "misses": 0,  # 缓存未命中次数
            "sets": 0,   # 缓存设置次数
            "evictions": 0,  # 缓存淘汰次数（LRU+过期）
        }

    # =========================
    # 内部方法
    # =========================

    def _generate_key(self, session_id: str, query: str) -> str:
        """
        对查询语句规范化后生成MD5哈希key（保证精确匹配）
        
        规范化规则：
        1. 转小写
        2. 去除首尾空格
        3. 多个空格合并为单个空格
        注意：MD5存在极低的哈希碰撞风险，生产环境可替换为SHA256
        """        
        normalized = " ".join(query.lower().strip().split())
        result = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return f"{session_id}:{result}"
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self.ttl

    # =========================
    # 核心接口
    # =========================

    def get(self, session_id: str, query: str) -> Optional[str]:
        """获取缓存结果"""
        
        with self._lock:
            key = self._generate_key(session_id,query)

            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                logger.debug(f"缓存未命中：{query}")
                return None

            if self._is_expired(entry):
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                logger.debug(f"缓存已过期：{query}")
                return None

            # 命中
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats["hits"] += 1

            logger.debug(
                f"缓存命中：{query}（命中次数：{entry.hit_count}）"
            )
            return entry.answer

    def set(
        self,
        session_id: str,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """设置缓存"""
        key = self._generate_key(session_id, query)

        with self._lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = CacheEntry(
                query=query,
                answer=answer,
                created_at=time.time(),
                metadata=metadata or {},
            )

            self._stats["sets"] += 1
            logger.debug(f"缓存已设置：{query}")

            # LRU 淘汰
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1
                logger.info("LRU 淘汰一个缓存条目")

            return key

    def invalidate(self,session_id: str, query: str) -> bool:
        """删除指定缓存"""
        key = self._generate_key(session_id, query)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"缓存已删除：{query}")
                return True
            return False

    def clear(self) -> int:
        """清空所有缓存"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"已清空缓存：{count} 条")
            return count

    def cleanup_expired(self) -> int:
        """清理过期缓存"""
        with self._lock:
            expired_keys = [
                key
                for key, entry in list(self._cache.items())
                if self._is_expired(entry)
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                self._stats["evictions"] += len(expired_keys)
                logger.info(f"清理过期缓存：{len(expired_keys)} 条")

            return len(expired_keys)

    # =========================
    # 统计 & 查询
    # =========================

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "total_requests": total,
                "hit_rate": round(hit_rate, 4),
                **self._stats,
            }

    def get_hot_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最热门查询"""
        with self._lock:
            entries = sorted(
                self._cache.values(),
                key=lambda e: e.hit_count,
                reverse=True,
            )[:limit]

            return [
                {"query": e.query, "hit_count": e.hit_count}
                for e in entries
            ]

    # =========================
    # 兼容旧接口
    # =========================

    def get_cached_response(self, session_id: str ,query: str) -> Optional[Dict[str, Any]]:
        result = self.get(session_id,query)
        if result is None:
            return None
        return {"answer": result}

    def cache_response(self, session_id: str, query: str, answer: str, **kwargs) -> str:
        return self.set(session_id, query, answer, kwargs.get("metadata"))


# =========================
# 全局实例
# =========================

_cache_manager: Optional[CacheManager] = None


def get_cache_manager(
    max_size: int = 1000,
    ttl: int = 3600,
    enable_semantic: bool = False
) -> CacheManager:
    """
    获取全局缓存管理器实例（单例模式）
    
    Args:
        max_size: 缓存最大条目数
        ttl: 缓存过期时间（秒）
        enable_semantic: 语义缓存开关（预留）
    Returns:
        全局单例CacheManager
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            max_size=max_size,
            ttl=ttl,
            enable_semantic=enable_semantic
        )
    return _cache_manager
            


if __name__ == "__main__":
    # 初始化缓存管理器（自定义配置）
    cache = get_cache_manager(max_size=2, ttl=30)
    
    # 设置缓存
    cache.set("123", "红烧肉怎么做？", "红烧肉做法：1.选肉 2.焯水...", metadata={"routing": "hybrid"})
    cache.set("124", "可乐鸡翅怎么做", "可乐鸡翅做法：1.腌制 2.煎制...")
    
    # 命中缓存
    print("命中缓存：", cache.get("123","红烧肉怎么做？"))  # 输出回答内容
    print("热门查询：", cache.get_hot_queries())      # 输出[{"query": "...", "hit_count": 1, ...}]
    
    # 触发LRU淘汰（超过max_size=2）
    cache.set("125","番茄炒蛋怎么做", "番茄炒蛋做法：...")
    print("淘汰后缓存数：", cache.get_stats()["size"])  # 输出2
    
    # 统计信息
    stats = cache.get_stats()
    print("命中率：", stats["hit_rate"])  # 输出1.0（1次命中，0次未命中）

        

