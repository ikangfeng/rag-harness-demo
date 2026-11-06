#%%
"""
主体程序实现：
1. 对金融QA数据集1.csv以及对金融doc数据集1.txt 进行知识库融合构建
2.按照需求完成多层级的RAG整个过程
3.优先在metadata中type=query的子集合中按照阈值进行高相似度检索,看能否直接命中
"""
# ====== 专门解决Matplotlib警告问题 ======
# 设置Matplotlib配置目录为可写目录
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
# 确保目录存在
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
# 禁用Matplotlib字体缓存构建
os.environ['MATPLOTLIB_FONT_CACHE'] = '/tmp/matplotlib_cache/fontlist-v330.json'
# 禁用Matplotlib的debug输出
os.environ['MPL_DEBUG'] = 'False'
# 禁用Matplotlib的后端检测
os.environ['MPLBACKEND'] = 'Agg'
# 禁用Matplotlib的样式加载
os.environ['MATPLOTLIBRC'] = '/dev/null'
# =====================================

import sys
import time
import logging
import os
from build_knowledge_vectorstore import init_vectordb,get_bm25_retriever,generation_replay_by_LLM,generation_replay_by_LLM_stream, llm

logger = logging.getLogger(__name__)

# ---- 加载 policy.yaml 里的路由阈值 ----
import yaml
_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.yaml")
with open(_POLICY_PATH, "r", encoding="utf-8") as f:
    _POLICY = yaml.safe_load(f)
_DIRECT_THRESHOLD = _POLICY["routing"]["direct_answer"]["threshold"]     # 0.5
_FALLBACK_THRESHOLD = _POLICY["routing"]["fallback"]["threshold"]          # 1.0
_FALLBACK_MESSAGE = _POLICY["routing"]["fallback"]["fallback_message"]
_QUERY_REWRITE_ENABLED = _POLICY["routing"]["query_rewrite"].get("enabled", True)

# 权限策略
_PERMISSIONS = _POLICY.get("permissions", {}).get("doc_access", {})


def rewrite_query(user_query: str) -> str:
    """
    Query Rewrite：把用户口语化查询改写为知识库标准问句。
    例如："查一下信用评级" → "什么是信用评级"
    """
    if not _QUERY_REWRITE_ENABLED:
        return user_query
    try:
        rewrite_prompt = f"""请将以下用户问题改写为金融知识库的标准问句格式（以"什么是X"/"解释X"/"X是什么"开头）。
只输出改写后的问句，不要有任何解释。

用户问题：{user_query}
改写后："""
        result = llm.invoke(rewrite_prompt).strip()
        # 只取第一行，去掉多余标点
        result = result.split("\n")[0].strip().rstrip("。.!！")
        if result:
            print(f"🔄 Query Rewrite: '{user_query}' → '{result}'")
            return result
    except Exception as e:
        print(f"⚠️ Query Rewrite 失败，使用原查询: {e}")
    return user_query


def filter_by_permission(docs_with_scores):
    """
    权限过滤：根据 policy.yaml 的 doc_access 策略过滤检索结果。
    internal/confidential 级别文档被过滤掉。
    """
    if not _PERMISSIONS:
        return docs_with_scores
    filtered = []
    for item in docs_with_scores:
        doc = item[0] if isinstance(item, tuple) else item
        access_level = getattr(doc, "metadata", {}).get("access_level", "public")
        if _PERMISSIONS.get(access_level, "allow") == "allow":
            filtered.append(item)
    return filtered

def safe_similarity_search(vs, query, k=3, max_retries=3, base_delay=1.0):
    """带重试的向量检索。embedding API 偶发 SSL 断连时自动重试，全部失败返回 [] 以降级到 BM25。"""
    for attempt in range(1, max_retries + 1):
        try:
            return vs.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.warning(f"向量检索第 {attempt}/{max_retries} 次失败: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(base_delay * attempt)  # 线性退避
    logger.error(f"向量检索连续 {max_retries} 次失败，降级到 BM25 检索")
    return []

# 延迟初始化向量库和检索器，只在第一次使用时初始化
vectorstore = None
high_resolution_retriever = None
middle_resolution_retriever = None
low_resolution_retriever = None
bm25_retriever = None

def init_retrievers():
    """延迟初始化向量库和检索器，只在第一次使用时初始化"""
    global vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever
    
    if vectorstore is None:
        print("初始化向量库...")
        vectorstore = init_vectordb()
        print("向量库初始化完成")
    
    if high_resolution_retriever is None:
        # 高精度检索器：用于直接命中精确匹配的查询
        high_resolution_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        
        # 中精度检索器：用于匹配相关的QA内容
        middle_resolution_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.68},
        )
        
        # 文档检索器：用于从文档中获取相关信息
        low_resolution_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3},
        )
        print("检索器初始化完成")
    
    if bm25_retriever is None:
        # BM25检索器：用于在中精度和自由生成时提供一定的其它召回机制的支持
        bm25_retriever = get_bm25_retriever(K=5)
        print("BM25检索器初始化完成")
    
    return vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever

def get_answer_from_doc(doc):
    """
    安全地从文档中获取答案
    优先从metadata['answer']获取，如果没有则从page_content中提取
    """
    # 如果metadata中有answer字段，直接返回
    if 'answer' in doc.metadata and doc.metadata['answer']:
        return doc.metadata['answer']
    
    # 否则从page_content中提取（qa类型的文档格式：问题答案：答案）
    page_content = doc.page_content
    if '答案：' in page_content:
        parts = page_content.split('答案：')
        if len(parts) >= 2:
            return parts[1].strip()
    
    # 如果都没有，返回整个page_content
    return page_content

#%%
#定义主函数,用于使用三个level的检索器解答用户提出的问题query
def answer_by_vectorstore(query,
                          history,
                          query_retriever=None,
                          qa_retriever=None,
                          doc_retriever=None,
                          bm25_retriever_param=None,
                          return_info=False):
    """
    :param query:用户本轮问题
    :param history: 历史记录
    :param query_retriever:基于query的检索器
    :param qa_retriever: 基于qa的检索器
    :param doc_retriever: 文档检索器
    :param bm25_retriever_param: 稀疏bm25检索器
    :param return_info: 是否返回路由决策和检索详情（供 Harness 使用）
    :return: answer 或 (answer, pipeline_info)
    """
    # 收集管道信息（供 Harness trace 使用）
    pipeline_info = {
        "routing_decision": {"branch": "unknown", "distance": None},
        "retrieved_docs": [],
    }
    # 延迟初始化向量库和检索器
    global vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever
    if vectorstore is None or high_resolution_retriever is None or middle_resolution_retriever is None or low_resolution_retriever is None or bm25_retriever is None:
        vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever = init_retrievers()
    
    # 如果函数参数没有提供，使用全局初始化的检索器
    if query_retriever is None:
        query_retriever = high_resolution_retriever
    if qa_retriever is None:
        qa_retriever = middle_resolution_retriever
    if doc_retriever is None:
        doc_retriever = low_resolution_retriever
    if bm25_retriever_param is None:
        bm25_retriever_param = bm25_retriever
    
    # 使用参数名bm25_retriever_param来避免与全局变量冲突
    bm25_retriever_result = bm25_retriever_param.invoke(query)
    
    # 预处理查询，确保与知识库中文档格式一致
    import re

    # Query Rewrite：口语化查询 → 标准问句
    original_query = query
    query = rewrite_query(query)
    pipeline_info["query_rewrite"] = {"original": original_query, "rewritten": query}

    # 移除查询末尾的标点符号和语气词
    processed_query = re.sub(r'[？?。.！!；;，,]$', '', query)
    processed_query = re.sub(r'[呢吗吧啊呀啦]$', '', processed_query)
    
    # 使用带距离的检索方法，查看原始检索结果和距离
    test_docs = safe_similarity_search(vectorstore,processed_query, k=3)
    # 权限过滤
    test_docs = filter_by_permission(test_docs)
    print(f"带距离的检索结果: {test_docs}")
    
    # 检查是否有直接命中的结果
    if test_docs:
        # 打印每个结果的详情，包括距离
        for i, (doc, distance) in enumerate(test_docs):
            print(f"结果 {i+1}: 距离={distance}, 内容={doc.page_content[:50]}..., 类型={doc.metadata['type']}")
        
        # 取距离最小的结果（最相似）
        best_doc, best_distance = test_docs[0]
        print(f"最佳匹配: 距离={best_distance}, 内容={best_doc.page_content[:50]}..., 答案={best_doc.metadata.get('answer', '无')}")
        
        # 如果距离足够小（相似性足够高），直接返回答案
        # 调整距离阈值为0.5，更严格的高精度匹配
        if best_distance < _DIRECT_THRESHOLD:  # 距离阈值调整为0.5，更严格的高精度匹配
            answer = get_answer_from_doc(best_doc)
            pipeline_info["routing_decision"] = {"branch": "direct_answer", "distance": float(best_distance), "method": "processed_query"}
            pipeline_info["retrieved_docs"] = [{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in test_docs]
            ans = f"{answer} (匹配距离: {best_distance:.4f})"
            return (ans, pipeline_info) if return_info else ans
    
    # 如果高精度匹配失败，尝试添加问号的版本（匹配知识库中文档格式）
    query_with_question_mark = processed_query + "？"
    test_docs_with_zh_question = filter_by_permission(safe_similarity_search(vectorstore,query_with_question_mark, k=3))
    if test_docs_with_zh_question:
        print(f"带中文问号的检索结果: {test_docs_with_zh_question}")
        best_doc, best_distance = test_docs_with_zh_question[0]
        if best_distance < _DIRECT_THRESHOLD:  # 使用统一的距离阈值0.5
            answer = get_answer_from_doc(best_doc)
            pipeline_info["routing_decision"] = {"branch": "direct_answer", "distance": float(best_distance), "method": "zh_question_mark"}
            pipeline_info["retrieved_docs"] = [{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in test_docs_with_zh_question]
            ans = f"{answer} (匹配距离: {best_distance:.4f})"
            return (ans, pipeline_info) if return_info else ans
    
    # 如果仍然失败，尝试英文问号
    query_with_question_mark_en = processed_query + "?"
    test_docs_with_en_question = filter_by_permission(safe_similarity_search(vectorstore,query_with_question_mark_en, k=3))
    if test_docs_with_en_question:
        print(f"带英文问号的检索结果: {test_docs_with_en_question}")
        best_doc, best_distance = test_docs_with_en_question[0]
        if best_distance < _DIRECT_THRESHOLD:  # 使用统一的距离阈值0.5
            answer = get_answer_from_doc(best_doc)
            pipeline_info["routing_decision"] = {"branch": "direct_answer", "distance": float(best_distance), "method": "en_question_mark"}
            pipeline_info["retrieved_docs"] = [{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in test_docs_with_en_question]
            ans = f"{answer} (匹配距离: {best_distance:.4f})"
            return (ans, pipeline_info) if return_info else ans
    
    # 最后尝试原始查询
    test_docs_with_original = filter_by_permission(safe_similarity_search(vectorstore,query, k=3))
    if test_docs_with_original:
        print(f"原始查询的检索结果: {test_docs_with_original}")
        best_doc, best_distance = test_docs_with_original[0]
        if best_distance < _DIRECT_THRESHOLD:  # 使用统一的距离阈值0.5
            answer = get_answer_from_doc(best_doc)
            pipeline_info["routing_decision"] = {"branch": "direct_answer", "distance": float(best_distance), "method": "original_query"}
            pipeline_info["retrieved_docs"] = [{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in test_docs_with_original]
            ans = f"{answer} (匹配距离: {best_distance:.4f})"
            return (ans, pipeline_info) if return_info else ans
    
    # === 兜底检查：如果最佳距离 > 阈值，说明完全超出知识库范围 ===
    all_low_res_results = filter_by_permission(safe_similarity_search(vectorstore, processed_query, k=1))
    if all_low_res_results and all_low_res_results[0][1] > _FALLBACK_THRESHOLD:
        print(f"⚠️ 兜底: 最佳距离={all_low_res_results[0][1]:.4f} > {_FALLBACK_THRESHOLD}, 超出知识库范围")
        pipeline_info["routing_decision"] = {"branch": "fallback", "distance": float(all_low_res_results[0][1])}
        pipeline_info["retrieved_docs"] = []
        return (_FALLBACK_MESSAGE, pipeline_info) if return_info else _FALLBACK_MESSAGE
    
    # 中精度匹配使用带距离的检索
    middle_resolution_results = safe_similarity_search(vectorstore,processed_query, k=5)
    # 权限过滤
    middle_resolution_results = filter_by_permission(middle_resolution_results)
    print(f"中精度检索结果: {middle_resolution_results}")
    
    # 打印BM25检索结果数量
    print(f"BM25检索结果数量: {len(bm25_retriever_result)}")
    
    # BM25结果去重 + 权限过滤
    unique_bm25_results = []
    seen_bm25_contents = set()
    for result in bm25_retriever_result:
        if result.page_content not in seen_bm25_contents:
            seen_bm25_contents.add(result.page_content)
            unique_bm25_results.append(result)
    # BM25 权限过滤（Document 没有分数，用 filter_by_permission 不兼容 tuple，单独过滤）
    if _PERMISSIONS:
        unique_bm25_results = [
            d for d in unique_bm25_results
            if _PERMISSIONS.get(getattr(d, "metadata", {}).get("access_level", "public"), "allow") == "allow"
        ]
    bm25_retriever_result = unique_bm25_results
    print(f"去重+权限过滤后BM25结果数量: {len(bm25_retriever_result)}")
    
    # === Rerank 精排（工业级）===
    # 多路召回 → 合并去重 → Rerank 精排 → Top3
    rerank_applied = False
    reranked_docs = []
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reranker import Reranker
        
        # 合并向量检索 + BM25 候选
        combined_candidates = []
        seen = set()
        for doc, dist in middle_resolution_results:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined_candidates.append({"content": doc.page_content, "metadata": doc.metadata})
        for doc in bm25_retriever_result:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined_candidates.append({"content": doc.page_content, "metadata": doc.metadata})
        
        print(f"Rerank 候选文档: {len(combined_candidates)}")
        if len(combined_candidates) > 1:
            reranker = Reranker()
            reranked, rerank_method = reranker.rerank(processed_query, combined_candidates, top_k=3)
            rerank_applied = True
            reranked_docs = reranked
            pipeline_info["rerank_method"] = rerank_method
            print(f"✅ Rerank 精排完成 [{rerank_method}]: {[(d['content'][:30], f'{s:.2f}') for d, s in reranked]}")
    except Exception as e:
        print(f"⚠️ Rerank 未启用（降级）: {e}")
    
    # 构建知识组合字符串，包含距离信息
    combine_knowledge_str = ""
    
    # 决定用哪些文档作为背景知识
    background_docs = reranked_docs if rerank_applied and reranked_docs else [(d, s) for d, s in middle_resolution_results[:3]]
    pipeline_info["rerank_applied"] = rerank_applied
    pipeline_info["rerank_top_docs"] = [
        {"content": d.get("content", "")[:100], "score": float(s)} for d, s in reranked_docs
    ] if rerank_applied and reranked_docs else []
    # P1#2 保存 before/after 用于对比展示
    if rerank_applied:
        pipeline_info["rerank_before_top"] = [
            {"content": d.page_content[:100], "score": float(s)} 
            for d, s in middle_resolution_results[:3]
        ]
        pipeline_info["rerank_after_top"] = [
            {"content": d.get("content", "")[:100], "score": float(s)} 
            for d, s in reranked_docs[:3]
        ]
    
    # 如果有中精度检索结果
    if background_docs:
        combine_knowledge_str += "=== 相关知识库匹配结果 ===\n"
        for i, item in enumerate(background_docs):
            doc_content = item[0].get("content") if rerank_applied else item[0].page_content
            score = item[1]
            score_label = "Rerank分" if rerank_applied else "距离"
            combine_knowledge_str += f"【匹配 {i+1}，{score_label}: {score:.4f}】\n"
            parts = doc_content.split("答案：")
            if len(parts) >= 2:
                combine_knowledge_str += f"问题: {parts[0]}\n答案：{parts[1]}\n"
            else:
                combine_knowledge_str += f"知识: {doc_content}\n"
            combine_knowledge_str += "\n"
    
    # 追加BM25检索结果
    if bm25_retriever_result:
        combine_knowledge_str += "=== 文档检索结果 ===\n"
        for i, doc in enumerate(bm25_retriever_result):
            combine_knowledge_str += f"【文档 {i+1}】\n"
            combine_knowledge_str += f"内容: {doc.page_content}\n\n"
    
    # 文档检索结果（带距离）- 作为兜底机制（加权限过滤）
    doc_retriever_results = filter_by_permission(safe_similarity_search(vectorstore,query, k=3))
    print(f"文档检索结果: {doc_retriever_results}")
    
    if doc_retriever_results:
        combine_knowledge_str += "=== 文档片段匹配 ===\n"
        for i, (doc, distance) in enumerate(doc_retriever_results):
            combine_knowledge_str += f"【片段 {i+1}，距离: {distance:.4f}】\n"
            combine_knowledge_str += f"内容: {doc.page_content}\n\n"
    
    # 真正的兜底机制：如果所有检索都没有结果
    if not combine_knowledge_str.strip():
        combine_knowledge_str = "=== 兜底回答 ===\n"
        combine_knowledge_str += "未找到直接匹配的知识库内容，将基于通用知识回答。\n"
        combine_knowledge_str += "您的问题可能超出了当前知识库的范围。\n"
    
    #对历史记录的处理：拼接为统一的字符串
    chat_history_combine_str = ""
    for his in history:
        if his["role"] == "user":
            chat_history_combine_str += "用户:" + his["content"] + "\n"
        else:
            chat_history_combine_str += "AI:" + his["content"] + "\n"
    
    #调用LLM基于知识检索结果、历史记录进行QA问答
    print("调用LLM生成回复...")
    response = generation_replay_by_LLM(
        background_knowledge=combine_knowledge_str,
        history_message=chat_history_combine_str,
        current_query=query
    )
    
    # 快速响应机制：对于无法直接回答的问题，先返回友好提示
    if not combine_knowledge_str.strip() or "未找到直接匹配" in combine_knowledge_str:
        pipeline_info["routing_decision"] = {"branch": "fallback", "distance": None, "method": "no_match"}
        ans = "抱歉，我无法预测未来的股票走势，包括明天茅台的涨跌。股票市场受到多种因素影响，建议您关注相关财经新闻和专业分析。"
        return (ans, pipeline_info) if return_info else ans
    
    # 记录中精度检索到的文档
    all_retrieved = []
    if middle_resolution_results:
        all_retrieved.extend([{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in middle_resolution_results])
    if doc_retriever_results:
        all_retrieved.extend([{"content": d.page_content[:100], "score": float(s), "type": d.metadata.get("type")} for d, s in doc_retriever_results])
    pipeline_info["retrieved_docs"] = all_retrieved
    pipeline_info["routing_decision"] = {"branch": "llm_generate", "distance": None, "method": "middle_resolution"}

    # 添加检索统计信息
    total_matches = len(middle_resolution_results) + len(doc_retriever_results) + len(bm25_retriever_result)
    response += f"\n\n=== 检索信息 ===\n"
    response += f"总匹配结果: {total_matches} 项\n"
    response += f"知识库匹配: {len(middle_resolution_results)} 项\n"
    response += f"文档片段: {len(doc_retriever_results)} 项\n"
    response += f"BM25检索: {len(bm25_retriever_result)} 项\n"
    
    return (response, pipeline_info) if return_info else response

# 流式版本的回答函数
def answer_by_vectorstore_stream(query, history):
    """
    流式版本的回答函数（旧版本，模拟流式）
    """
    try:
        # 调用原始函数获取完整响应
        full_response = answer_by_vectorstore(query, history)
        
        # 逐字输出完整响应（模拟流式）
        for char in full_response:
            yield char
    except Exception as e:
        # 返回友好的错误提示
        yield f"抱歉，处理请求时出现错误：{str(e)}"

# 真正的流式版本的回答函数
def answer_by_vectorstore_stream_real(query, history):
    """
    真正的流式版本回答函数，使用LLM的流式API实时生成token
    """
    try:
        # 延迟初始化向量库和检索器
        global vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever
        if vectorstore is None or high_resolution_retriever is None or middle_resolution_retriever is None or low_resolution_retriever is None or bm25_retriever is None:
            vectorstore, high_resolution_retriever, middle_resolution_retriever, low_resolution_retriever, bm25_retriever = init_retrievers()
        
        # 预处理查询
        import re
        processed_query = re.sub(r'[？?。.！!；;，,]$', '', query)
        processed_query = re.sub(r'[呢吗吧啊呀啦]$', '', processed_query)
        
        # BM25检索
        bm25_retriever_result = bm25_retriever.invoke(query)
        
        # 检查是否有直接命中的结果
        test_docs = safe_similarity_search(vectorstore,processed_query, k=3)
        
        # 在终端输出检索详情（包括距离和文本块），强制刷新输出
        import sys
        print("\n" + "="*60, flush=True)
        print("=== 检索详情 ===", flush=True)
        if test_docs:
            print(f"【高精度检索结果 - Top {len(test_docs)}】", flush=True)
            for i, (doc, distance) in enumerate(test_docs):
                print(f"\n结果 {i+1}:", flush=True)
                print(f"  欧式距离: {distance:.4f}", flush=True)
                print(f"  类型: {doc.metadata.get('type', 'unknown')}", flush=True)
                print(f"  内容预览: {doc.page_content[:150]}...", flush=True)
                answer = get_answer_from_doc(doc)
                if answer and answer != doc.page_content:
                    print(f"  答案: {answer[:150]}...", flush=True)
            
            best_doc, best_distance = test_docs[0]
            print(f"\n最佳匹配: 距离={best_distance:.4f}", flush=True)
            if best_distance < _DIRECT_THRESHOLD:
                # 直接命中，流式返回答案
                answer = get_answer_from_doc(best_doc)
                answer_text = f"{answer} (匹配距离: {best_distance:.4f})"
                for char in answer_text:
                    yield char
                return
        
        # 尝试添加问号的版本
        query_with_question_mark = processed_query + "？"
        test_docs_with_zh_question = safe_similarity_search(vectorstore,query_with_question_mark, k=3)
        if test_docs_with_zh_question:
            best_doc, best_distance = test_docs_with_zh_question[0]
            if best_distance < _DIRECT_THRESHOLD:
                answer = get_answer_from_doc(best_doc)
                answer_text = f"{answer} (匹配距离: {best_distance:.4f})"
                for char in answer_text:
                    yield char
                return
        
        # 尝试英文问号
        query_with_question_mark_en = processed_query + "?"
        test_docs_with_en_question = safe_similarity_search(vectorstore,query_with_question_mark_en, k=3)
        if test_docs_with_en_question:
            best_doc, best_distance = test_docs_with_en_question[0]
            if best_distance < _DIRECT_THRESHOLD:
                answer = get_answer_from_doc(best_doc)
                answer_text = f"{answer} (匹配距离: {best_distance:.4f})"
                for char in answer_text:
                    yield char
                return
        
        # 尝试原始查询
        test_docs_with_original = safe_similarity_search(vectorstore,query, k=3)
        if test_docs_with_original:
            best_doc, best_distance = test_docs_with_original[0]
            if best_distance < _DIRECT_THRESHOLD:
                answer = get_answer_from_doc(best_doc)
                answer_text = f"{answer} (匹配距离: {best_distance:.4f})"
                for char in answer_text:
                    yield char
                return
        
        # 中精度匹配
        middle_resolution_results = safe_similarity_search(vectorstore,processed_query, k=5)
        
        # BM25结果去重
        unique_bm25_results = []
        seen_bm25_contents = set()
        for result in bm25_retriever_result:
            if result.page_content not in seen_bm25_contents:
                seen_bm25_contents.add(result.page_content)
                unique_bm25_results.append(result)
        bm25_retriever_result = unique_bm25_results
        
        # 文档检索结果
        doc_retriever_results = safe_similarity_search(vectorstore,query, k=3)
        
        # 在终端输出详细的检索信息（包括距离和文本块），强制刷新输出
        import sys
        print("\n" + "="*60, flush=True)
        print("=== 检索详情 ===", flush=True)
        
        if test_docs:
            print(f"\n【高精度检索结果 - Top {len(test_docs)}】", flush=True)
            for i, (doc, distance) in enumerate(test_docs):
                print(f"\n结果 {i+1}:", flush=True)
                print(f"  欧式距离: {distance:.4f}", flush=True)
                print(f"  类型: {doc.metadata.get('type', 'unknown')}", flush=True)
                print(f"  内容: {doc.page_content[:200]}", flush=True)
                answer = get_answer_from_doc(doc)
                if answer and answer != doc.page_content:
                    print(f"  答案: {answer[:200]}", flush=True)
        
        if middle_resolution_results:
            print(f"\n【中精度检索结果 - Top {len(middle_resolution_results)}】", flush=True)
            for i, (doc, distance) in enumerate(middle_resolution_results):
                print(f"\n匹配 {i+1}:", flush=True)
                print(f"  欧式距离: {distance:.4f}", flush=True)
                print(f"  类型: {doc.metadata.get('type', 'unknown')}", flush=True)
                parts = doc.page_content.split("答案：")
                if len(parts) >= 2:
                    print(f"  问题: {parts[0][:150]}", flush=True)
                    print(f"  答案: {parts[1][:150]}", flush=True)
                else:
                    print(f"  内容: {doc.page_content[:200]}", flush=True)
        
        if doc_retriever_results:
            print(f"\n【文档片段检索结果 - Top {len(doc_retriever_results)}】", flush=True)
            for i, (doc, distance) in enumerate(doc_retriever_results):
                print(f"\n片段 {i+1}:", flush=True)
                print(f"  欧式距离: {distance:.4f}", flush=True)
                print(f"  类型: {doc.metadata.get('type', 'unknown')}", flush=True)
                print(f"  内容: {doc.page_content[:200]}", flush=True)
        
        if bm25_retriever_result:
            print(f"\n【BM25检索结果 - Top {len(bm25_retriever_result)}】", flush=True)
            for i, doc in enumerate(bm25_retriever_result):
                print(f"\n文档 {i+1}:", flush=True)
                print(f"  类型: {doc.metadata.get('type', 'unknown')}", flush=True)
                print(f"  内容: {doc.page_content[:200]}", flush=True)
        
        # 输出检索统计
        total_matches = len(middle_resolution_results) + len(doc_retriever_results) + len(bm25_retriever_result)
        print(f"\n【检索统计】", flush=True)
        print(f"  总匹配结果: {total_matches} 项", flush=True)
        print(f"    - 知识库匹配: {len(middle_resolution_results)} 项", flush=True)
        print(f"    - 文档片段: {len(doc_retriever_results)} 项", flush=True)
        print(f"    - BM25检索: {len(bm25_retriever_result)} 项", flush=True)
        print("="*60 + "\n", flush=True)
        
        # 构建知识组合字符串（用于LLM生成）
        combine_knowledge_str = ""
        
        if middle_resolution_results:
            combine_knowledge_str += "=== 相关知识库匹配结果 ===\n"
            for i, (doc, distance) in enumerate(middle_resolution_results):
                combine_knowledge_str += f"【匹配 {i+1}，距离: {distance:.4f}】\n"
                parts = doc.page_content.split("答案：")
                if len(parts) >= 2:
                    combine_knowledge_str += f"问题: {parts[0]}\n答案：{parts[1]}\n"
                else:
                    combine_knowledge_str += f"知识: {doc.page_content}\n"
                combine_knowledge_str += "\n"
        
        if bm25_retriever_result:
            combine_knowledge_str += "=== 文档检索结果 ===\n"
            for i, doc in enumerate(bm25_retriever_result):
                combine_knowledge_str += f"【文档 {i+1}】\n"
                combine_knowledge_str += f"内容: {doc.page_content}\n\n"
        
        if doc_retriever_results:
            combine_knowledge_str += "=== 文档片段匹配 ===\n"
            for i, (doc, distance) in enumerate(doc_retriever_results):
                combine_knowledge_str += f"【片段 {i+1}，距离: {distance:.4f}】\n"
                combine_knowledge_str += f"内容: {doc.page_content}\n\n"
        
        # 兜底机制
        if not combine_knowledge_str.strip():
            combine_knowledge_str = "=== 兜底回答 ===\n"
            combine_knowledge_str += "未找到直接匹配的知识库内容，将基于通用知识回答。\n"
            combine_knowledge_str += "您的问题可能超出了当前知识库的范围。\n"
        
        # 处理历史记录
        chat_history_combine_str = ""
        for his in history:
            if his["role"] == "user":
                chat_history_combine_str += "用户:" + his["content"] + "\n"
            else:
                chat_history_combine_str += "AI:" + his["content"] + "\n"
        
        # 快速响应机制
        if not combine_knowledge_str.strip() or "未找到直接匹配" in combine_knowledge_str:
            error_msg = "抱歉，我无法预测未来的股票走势，包括明天茅台的涨跌。股票市场受到多种因素影响，建议您关注相关财经新闻和专业分析。"
            for char in error_msg:
                yield char
            return
        
        # 调用流式LLM生成回复
        print("调用流式LLM生成回复...", flush=True)
        
        accumulated_response = ""
        for chunk in generation_replay_by_LLM_stream(
            background_knowledge=combine_knowledge_str,
            history_message=chat_history_combine_str,
            current_query=query
        ):
            accumulated_response += chunk
            yield chunk
        
        # 在终端输出最终回答
        print("\n" + "="*60, flush=True)
        print("【AI回答】", flush=True)
        print(accumulated_response, flush=True)
        print("="*60 + "\n", flush=True)
        
        # 自动生成评估报告
        try:
            from auto_evaluation import generate_evaluation_report
            
            # 收集所有检索到的文档和分数
            all_retrieved_docs = []
            
            # 添加高精度检索结果
            if test_docs:
                all_retrieved_docs.extend(test_docs)
            
            # 添加中精度检索结果
            if middle_resolution_results:
                all_retrieved_docs.extend(middle_resolution_results)
            
            # 添加文档检索结果
            if doc_retriever_results:
                all_retrieved_docs.extend(doc_retriever_results)
            
            # 去重（基于文档内容）
            seen_contents = set()
            unique_docs = []
            for doc, score in all_retrieved_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append((doc, score))
            
            # 按分数排序，取Top 5
            unique_docs.sort(key=lambda x: x[1])
            top_docs = unique_docs[:5]
            
            # 清理生成的答案（移除统计信息）
            clean_answer = accumulated_response
            if "=== 检索信息 ===" in clean_answer:
                clean_answer = clean_answer.split("=== 检索信息 ===")[0].strip()
            
            # 生成评估报告（自动从知识库查找标准答案）
            report_path = generate_evaluation_report(
                query=query,
                retrieved_docs_with_scores=top_docs,
                generated_answer=clean_answer,
                ground_truth_answer=None,  # 会自动从知识库查找
                vectorstore=vectorstore  # 传入vectorstore用于查找标准答案
            )
            
            print(f"\n✅ 评估报告已自动生成: {report_path}", flush=True)
            print(f"   查看报告: cat {report_path}", flush=True)
            
        except Exception as e:
            print(f"\n⚠️ 生成评估报告时出错: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        # 返回友好的错误提示
        error_msg = f"抱歉，处理请求时出现错误：{str(e)}"
        for char in error_msg:
            yield char
#%%
# query="第2季度GDP增速2.4%但贫富分化加剧基尼系数是多少"
# query="你好"
# query="什么是股票期权？" #股票期权是一种赋予持有者在未来某一日期以特定价格买卖股票的权利的金融衍生品。
# history=[{"role":"user","content":"你好"},{"role":"assistant","content":"请问有什么能帮到您的呢？"}]
# response=answer_by_vectorstore(query=query,history=history)
# print("response:", response)