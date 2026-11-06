"""
Rerank 重排序模块
================
三种 rerank 方案，自动降级：
  1. DashScope gte-rerank API（如果账号有开通）
  2. LLM Rerank（tongyi/qwen-max 对候选文档打分排序）
  3. 降级：返回原始向量检索结果

工业级 RAG 标准架构：多路召回（Top10~20）→ Rerank 精排 → Top3
"""

import os
import json
import re
import time
from typing import List, Tuple


class Reranker:
    def __init__(self, api_key: str = None):
        if api_key is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from config import rag_test_Config
                api_key = rag_test_Config().api_key
            except Exception:
                api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.api_key = api_key

    def rerank(
        self, query: str, docs: List[dict], top_k: int = 3
    ) -> tuple:
        """重排序入口，自动尝试可用方案
        返回: (results: List[Tuple[dict, float]], method: str)
        method: "gte-rerank" | "llm-rerank" | "fallback"
        """
        if not docs:
            return [], "none"

        # 方案 1: DashScope Rerank API
        results = self._rerank_via_api(query, docs, top_k)
        if results is not None:
            return results, "gte-rerank"

        # 方案 2: LLM Rerank
        results = self._rerank_via_llm(query, docs, top_k)
        if results is not None:
            return results, "llm-rerank"

        # 方案 3: 降级（返回原始 docs）
        print(f"⚠️ Rerank 全部降级，返回向量检索原始顺序")
        return [(d, 0.0) for d in docs[:top_k]], "fallback"

    def _rerank_via_api(self, query, docs, top_k):
        """DashScope gte-rerank API"""
        import urllib.request
        body = json.dumps({
            "model": "gte-rerank",
            "input": {"query": query, "documents": [d["content"] for d in docs]},
            "parameters": {"top_n": min(top_k, len(docs))},
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/generate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            results = result.get("output", {}).get("results", [])
            return [(docs[item["index"]], item["relevance_score"]) for item in results]
        except Exception:
            return None

    def _rerank_via_llm(self, query, docs, top_k):
        """LLM 打分排序（用 qwen-max 对每个 doc 打 0~10 相关性分）"""
        try:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            # 确保 API Key 已设置（build_knowledge_vectorstore.py 会设环境变量）
            from config import rag_test_Config
            os.environ["DASHSCOPE_API_KEY"] = rag_test_Config().api_key

            from langchain_community.llms import Tongyi
            llm = Tongyi(model_name="qwen-max", temperature=0.0)

            docs_text = "\n".join(f"[{i}] {d['content'][:120]}" for i, d in enumerate(docs))
            rerank_prompt = f"""你是信息检索专家。请评估以下 {len(docs)} 个文档与用户问题的相关性。
对每个文档打 0~10 分（0=完全无关，10=完美匹配）。

用户问题: {query}

候选文档:
{docs_text}

输出要求：每行 [序号]: [分数] ，最后一行: TOP_K=[最相关的 {top_k} 个序号用逗号分隔]"""

            result = llm.invoke(rerank_prompt)

            scores = {}
            for line in result.split("\n"):
                m = re.match(r"\[(\d+)\].*?(\d+(?:\.\d+)?)", line)
                if m:
                    scores[int(m.group(1))] = float(m.group(2))

            top_line = re.search(r"TOP_K=([\d,]+)", result)
            if top_line:
                indices = [int(x.strip()) for x in top_line.group(1).split(",") if x.strip()]
            else:
                indices = sorted(scores.keys(), key=lambda i: -scores.get(i, 0))[:top_k]

            return [(docs[i], scores.get(i, 0)) for i in indices]

        except Exception as e:
            print(f"  LLM Rerank 失败: {e}")
            return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import os
    os.environ["RAG_VECTORSTORE_DIR"] = "/Users/feng/Documents/trae_projects/zjj/project2_vectorstore"

    reranker = Reranker()
    query = "什么是股票期权"
    docs = [
        {"content": "股票期权是一种赋予持有者在未来某一日期以特定价格买卖股票的权利的金融衍生品。"},
        {"content": "什么是期权定价模型"},
        {"content": "Black-Scholes 模型是一种期权定价模型"},
        {"content": "股票市场概述"},
    ]

    print(f"Query: {query}\n{'─'*50}")
    results = reranker.rerank(query, docs, top_k=4)
    for doc, score in results:
        print(f"  {score:6.2f} | {doc['content'][:50]}")
