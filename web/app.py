"""
C9 图RAG系统 - Flask 后端

=== 这个文件是做什么的？ ===
这是Web服务器的后端代码，负责：
1. 提供网页界面（index.html）
2. 处理前端发来的API请求
3. 调用RAG系统回答问题

=== 前后端交互流程 ===
浏览器(前端)                    Flask服务器(后端)
    │                                   │
    │ ──── 1. 打开网页 ──────────────→ │  返回HTML页面
    │ ──── 2. 点击初始化 ─────────────→ │  加载模型、连接数据库
    │ ──── 3. 发送问题 ───────────────→ │  检索+生成回答
    │ ←─── 4. 返回回答 ─────────────── │

=== API接口说明 ===
- GET  /              : 返回网页界面
- GET  /api/status    : 检查系统状态
- POST /api/init      : 初始化RAG系统
- POST /api/ask       : 提问并获取回答
- GET  /api/ask_stream: 流式问答（SSE）
- POST /api/agent     : Agent模式问答
- GET  /api/session   : 会话管理
- GET  /api/cache/stats: 缓存统计
"""
import os
import sys
import json
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)



# ==================== 路径设置 ====================
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, jsonify, send_from_directory, request, Response, stream_with_context

# ==================== 创建Flask应用 ====================
app = Flask(__name__, static_folder='static')

# ==================== 全局变量 ====================
rag_system = None

# ==================== 路由定义 ====================
# @app.route() 是装饰器，告诉Flask这个函数处理哪个URL

@app.route('/')
def index():
    """
    首页路由
    
    当用户访问 http://127.0.0.1:5000/ 时，返回index.html页面
    send_from_directory 从static目录读取并返回文件
    """
    return send_from_directory('static', 'index.html')

@app.route('/api/status')
def get_status():
    """
    获取系统状态
    
    前端会定期调用这个接口检查系统是否已初始化
    返回JSON格式：{'initialized': true/false, 'message': '...'}
    """
    global rag_system
    print(f"[DEBUG] 状态检查 - rag_system: {rag_system is not None}")
    return jsonify({
        'initialized': rag_system is not None,
        'message': '系统已就绪' if rag_system else '系统未初始化'
    })

@app.route('/api/init', methods=['POST'])
def init_system():
    """
    初始化RAG系统
    
    这个过程比较慢（通常需要10-30秒），包括：
    1. 连接Neo4j图数据库
    2. 连接Milvus向量数据库
    3. 加载AI嵌入模型（把文字变向量）
    4. 加载知识库数据
    5. 初始化检索引擎
    
    POST请求：前端点击"初始化"按钮时调用
    """
    global rag_system
    
    # 如果已经初始化过，直接返回成功
    if rag_system is not None:
        return jsonify({'success': True, 'message': '系统已初始化'})
    
    try:
        print("正在初始化系统...")
        
        # 导入配置和主程序（在需要时才导入，避免启动时就加载所有依赖）
        from config import DEFAULT_CONFIG
        from main import RecipeRAGSystem
        
        # 创建RAG系统实例
        rag_system = RecipeRAGSystem()
        
        # 初始化所有模块（连接数据库、加载模型等）
        rag_system.initialize_system()
        
        # 构建或加载知识库
        rag_system.build_knowledge_base()
        
        print("✅ 系统初始化完成")
        return jsonify({'success': True, 'message': '初始化成功'})
        
    except Exception as e:
        error_msg = f"初始化失败: {str(e)}"
        logger.error(error_msg, exc_info=True)  # 记录详细堆栈
        return jsonify({'success': False, 'message': error_msg})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    处理问答请求
    
    接收用户问题，调用RAG系统生成回答
    
    请求格式（JSON）：{'question': '红烧肉怎么做', 'session_id': 'xxx', 'use_cache': true}
    响应格式（JSON）：{
        'success': true,
        'answer': '红烧肉的做法是...',
        'routing': {'strategy': 'hybrid_traditional', 'confidence': 0.9},
        'cache_hit': false
    }
    """
    global rag_system, session_manager, cache_manager
    
    print(f"[DEBUG] 问答请求 - rag_system: {rag_system is not None}")
    
    # 检查系统是否已初始化
    if rag_system is None:
        return jsonify({'success': False, 'answer': '', 'message': '系统未就绪，请先构建知识库'})
    
    # 从请求中获取问题
    data = request.get_json()  # 解析JSON请求体
    question = data.get('question', '').strip()  # 获取问题并去除首尾空格
    session_id = data.get('session_id')
    use_cache = data.get('use_cache', True)
    
    if not question:
        return jsonify({'success': False, 'answer': '', 'message': '请输入问题'})
    
    try:
        print(f"问题: {question}")
        
        # 核心逻辑：调用RAG系统回答问题
        answer = rag_system.ask_question(question=question, session_id=session_id, stream=False)
        
        return jsonify({
            'success': True, 
            'answer': answer, 
            'cache_hit': False  # 在main.py中处理缓存，这里返回false
        })
        
    except Exception as e:
        print(f"❌ 问答失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'answer': '', 'message': str(e)})


# ==================== 流式输出 ====================
@app.route('/api/ask/stream', methods=['GET'])
def ask_stream():
    """
    流式问答接口（SSE - Server-Sent Events）
    
    使用GET请求，通过URL参数传递：
    - question: 用户问题
    - session_id: 会话ID（可选）
    - use_cache: 是否使用缓存（可选）
    
    返回SSE流，每个事件格式：
    data: {"type": "token", "content": "..."}
    data: {"type": "done", "cache_hit": false}
    """
    global rag_system, cache_manager
    
    question = request.args.get('question', '').strip()
    session_id = request.args.get('session_id')
    use_cache = request.args.get('use_cache', 'true').lower() == 'true'
    
    if not question:
        return Response(
            'data: {"type": "error", "message": "请输入问题"}\n\n',
            mimetype='text/event-stream'
        )
    
    if rag_system is None:
        return Response(
            'data: {"type": "error", "message": "系统未初始化"}\n\n',
            mimetype='text/event-stream'
        )
    
    def generate():
        try:
            # 调用RAG系统（这里简化为非流式，然后逐字符输出）
            # 真正的流式需要LLM支持流式生成
            response_generator = rag_system.ask_question(question=question, session_id=session_id, stream=True)
                        # 直接迭代生成器返回的内容
            for chunk in response_generator:
                yield f'data: {json.dumps({"type": "token", "content": chunk})}\n\n'
            
            # 流式结束标识
            yield f'data: {json.dumps({"type": "done", "cache_hit": False})}\n\n'
            
        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# ==================== 会话管理 ====================


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """创建新会话"""
    global rag_system
    
    if not rag_system:
        return jsonify({'success': False, 'message': '系统未初始化'})
  
    session_id = rag_system.session_manager.create_session()
    return jsonify({'success': True, 'session_id': session_id})

@app.route('/api/session/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """获取会话历史"""
    global rag_system
    
    if rag_system is None:
        return jsonify({'success': False, 'message': '系统为初始化', 'history': []})
    
    history = rag_system.session_manager.get_messages(session_id)
    return jsonify({'success': True, 'history': history})

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除会话"""
    global rag_system
    
    if rag_system is not None:
        rag_system.session_manager.delete_session(session_id)
    
    return jsonify({'success': True})





# ==================== 缓存统计 ====================
@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """获取缓存统计信息"""
    global rag_system
    
    if rag_system is None:
        return jsonify({'success': False, 'message': '系统未初始化'})
    
    stats = rag_system.cache_manager.get_stats()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """清空缓存"""
    global rag_system
    
    if rag_system:
        rag_system.cache_manager.clear()
    
    return jsonify({'success': True, 'message': '缓存已清空'})


if __name__ == '__main__':
    # 启动时初始化可选组件
    # init_session_manager()
    # init_cache_manager()
    
    # 启动Flask服务器
    # host='127.0.0.1' 只允许本地访问
    # port=5000 监听5000端口
    # debug=False 生产模式（不自动重载）
    print("启动服务: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)