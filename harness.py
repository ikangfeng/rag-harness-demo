"""
Harness 约束层
=============
借鉴 Claude Code 的 Policy-as-Code 思路，适配 RAG 场景。
职责：
  1. 前置安全检查（prompt injection 检测）
  2. 权限过滤（文档 ACL）
  3. 后置 PII 脱敏
  4. 全链路 trace 审计

设计原则：策略与执行分离，策略从 policy.yaml 读取，不写死在代码里。
"""

import os
import re
import json
import time
import uuid
import yaml
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 策略加载
# ============================================================
class PolicyLoader:
    """加载 policy.yaml 声明式策略文件"""

    def __init__(self, policy_path: Optional[str] = None):
        if policy_path is None:
            policy_path = os.path.join(os.path.dirname(__file__), "policy.yaml")
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)

    @property
    def permissions(self):
        return self.policy["permissions"]

    @property
    def security(self):
        return self.policy["security"]

    @property
    def routing(self):
        return self.policy["routing"]


# ============================================================
# 安全检查：Prompt Injection 检测
# ============================================================
class SecurityChecker:
    """检测 prompt injection，命中则 deny"""

    def __init__(self, policy: PolicyLoader):
        self.injection_config = policy.security["prompt_injection"]
        self.patterns = self.injection_config["patterns"]
        self.deny_message = self.injection_config["deny_message"]

    def check(self, query: str) -> dict:
        """
        返回: {"passed": bool, "reason": str, "matched_pattern": str|None}
        """
        for pattern in self.patterns:
            if pattern.lower() in query.lower():
                return {
                    "passed": False,
                    "reason": f"命中注入模式: {pattern}",
                    "matched_pattern": pattern,
                }
        return {"passed": True, "reason": "通过安全检查", "matched_pattern": None}


# ============================================================
# 权限过滤：文档 ACL
# ============================================================
class PermissionFilter:
    """
    根据文档的 access_level metadata 过滤检索结果。
    当前为单用户演示版：只允许 public，internal/confidential 直接 deny。
    """

    def __init__(self, policy: PolicyLoader):
        self.doc_access = policy.permissions["doc_access"]

    def filter_docs(self, docs_with_scores: list) -> tuple:
        """
        过滤检索结果，只保留有权限访问的文档。
        docs_with_scores: [(doc, score), ...]
        返回: (filtered_docs, denied_count)
        """
        filtered = []
        denied_count = 0
        for doc, score in docs_with_scores:
            access_level = doc.metadata.get("access_level", "public")
            if self.doc_access.get(access_level, "deny") == "allow":
                filtered.append((doc, score))
            else:
                denied_count += 1
        return filtered, denied_count


# ============================================================
# PII 脱敏
# ============================================================
class PIISanitizer:
    """对输出中的 PII（手机号、身份证、邮箱、银行卡）进行脱敏"""

    def __init__(self, policy: PolicyLoader):
        self.types = policy.security["pii_redaction"]["types"]
        # 预编译正则
        self.compiled = {
            name: re.compile(config["pattern"])
            for name, config in self.types.items()
        }

    def sanitize(self, text: str) -> tuple:
        """
        返回: (sanitized_text, redaction_log)
        redaction_log: [{"type": "phone", "count": 2}, ...]
        """
        sanitized = text
        redaction_log = []
        for name, regex in self.compiled.items():
            replacement = self.types[name]["replacement"]
            matches = regex.findall(sanitized)
            if matches:
                sanitized = regex.sub(replacement, sanitized)
                redaction_log.append({"type": name, "count": len(matches)})
        return sanitized, redaction_log


# ============================================================
# Trace 审计
# ============================================================
@dataclass
class TraceRecord:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    query: str = ""
    security_check: dict = field(default_factory=dict)
    permission_filter: dict = field(default_factory=dict)
    routing_decision: dict = field(default_factory=dict)
    retrieved_docs: list = field(default_factory=list)
    query_rewrite: dict = field(default_factory=dict)
    rerank_info: dict = field(default_factory=dict)
    query_pii_cleaned: list = field(default_factory=list)
    pii_redaction: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    final_answer: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "security_check": self.security_check,
            "permission_filter": self.permission_filter,
            "routing_decision": self.routing_decision,
            "retrieved_docs": self.retrieved_docs,
            "query_rewrite": self.query_rewrite,
            "rerank_info": self.rerank_info,
            "cost": self.cost,
            "pii_redaction": self.pii_redaction,
            "final_answer": self.final_answer[:200],
            "error": self.error,
        }


# ============================================================
# Harness 主类
# ============================================================
class Harness:
    """
    Harness 约束层主类。
    包裹 RAG 管道，在调用前后插入安全检查、权限过滤、PII 脱敏和 trace 审计。
    新增：缓存层 + 成本统计
    """

    # DashScope 定价（元/千 tokens，近似值，用于演示）
    PRICING = {
        "embedding": 0.0007,   # text-embedding-v3
        "llm_input": 0.008,    # qwen-max 输入
        "llm_output": 0.008,   # qwen-max 输出
        "rerank_llm": 0.008,   # LLM Rerank
    }

    def __init__(self, policy_path: Optional[str] = None):
        self.policy = PolicyLoader(policy_path)
        self.security_checker = SecurityChecker(self.policy)
        self.permission_filter = PermissionFilter(self.policy)
        self.pii_sanitizer = PIISanitizer(self.policy)
        self.trace_store = []  # 内存存储 trace（演示版）
        # 缓存层：{query: (answer, trace_info, timestamp)}，TTL 3600s
        self._cache = {}
        self._cache_ttl = 3600
        self._cache_hits = 0
        self._cache_misses = 0

    def run(self, query: str, rag_pipeline_func, harness_enabled: bool = True) -> tuple:
        """
        执行带约束的 RAG 查询。
        harness_enabled=False 时跳过所有 Harness 检查（用于对比演示）
        rag_pipeline_func: 接受 query，返回 (answer, trace_info) 的函数
        返回: (answer, trace)
        """
        import time
        trace = TraceRecord(query=query)

        # P2#1 无约束模式：跳过所有检查，直接跑管道
        if not harness_enabled:
            trace.security_check = {"passed": True, "reason": "Harness 已禁用（无约束模式）", "matched_pattern": None}
            trace.permission_filter = {"applied": False, "reason": "Harness 已禁用"}
            try:
                answer, pipeline_info = rag_pipeline_func(query)
                if not isinstance(pipeline_info, dict):
                    pipeline_info = {}
            except Exception as e:
                answer = f"错误: {e}"
                pipeline_info = {"error": str(e)}
            trace.routing_decision = pipeline_info.get("routing_decision", {"branch": "unknown", "distance": None})
            trace.retrieved_docs = pipeline_info.get("retrieved_docs", [])
            trace.rerank_info = {"applied": pipeline_info.get("rerank_applied", False), "method": pipeline_info.get("rerank_method", "none")}
            trace.cost = {"tokens": 50, "cost_yuan": 0.00004, "cache_hit": False}
            trace.final_answer = answer
            trace.harness_bypassed = True
            self._save_trace(trace)
            return answer, trace

        # ---- 前置：安全检查 ----
        sec_result = self.security_checker.check(query)
        trace.security_check = sec_result
        if not sec_result["passed"]:
            trace.final_answer = self.security_checker.deny_message
            trace.cost = {"tokens": 0, "cost_yuan": 0.0, "cache_hit": False}
            self._save_trace(trace)
            return self.security_checker.deny_message, trace

        # ---- 前置：清理 query 中的 PII（让检索能正常命中） ----
        clean_query, query_pii = self.pii_sanitizer.sanitize(query)
        if query_pii:
            print(f"🔧 前置清理 query 中的 PII: {query_pii}")
            trace.query_pii_cleaned = query_pii
            query = clean_query

        # ---- 缓存层检查 ----
        cache_key = query.strip()
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[2]) < self._cache_ttl:
            self._cache_hits += 1
            cached_answer, cached_info, _ = cached
            trace.routing_decision = cached_info.get("routing_decision", {})
            trace.retrieved_docs = cached_info.get("retrieved_docs", [])
            trace.query_rewrite = cached_info.get("query_rewrite", {})
            trace.rerank_info = {
                "applied": cached_info.get("rerank_applied", False),
                "method": cached_info.get("rerank_method", "cache"),
                "top_docs": cached_info.get("rerank_top_docs", []),
            }
            trace.cost = {"tokens": 0, "cost_yuan": 0.0, "cache_hit": True}
            trace.final_answer = cached_answer
            print(f"⚡ 缓存命中: {query[:30]}")
            self._save_trace(trace)
            return cached_answer, trace

        self._cache_misses += 1

        # ---- 调用 RAG 管道 ----
        try:
            answer, pipeline_info = rag_pipeline_func(query)
        except Exception as e:
            trace.error = str(e)
            answer = f"抱歉，处理请求时出现错误：{str(e)}"
            pipeline_info = {}

        # 记录管道信息
        trace.routing_decision = pipeline_info.get("routing_decision", {})
        trace.retrieved_docs = pipeline_info.get("retrieved_docs", [])
        trace.query_rewrite = pipeline_info.get("query_rewrite", {})
        trace.rerank_info = {
            "applied": pipeline_info.get("rerank_applied", False),
            "method": pipeline_info.get("rerank_method", "none"),
            "top_docs": pipeline_info.get("rerank_top_docs", []),
            "before_top": pipeline_info.get("rerank_before_top", []),
            "after_top": pipeline_info.get("rerank_after_top", []),
        }

        # ---- 成本统计 ----
        branch = trace.routing_decision.get("branch", "unknown")
        tokens, cost = self._estimate_cost(branch, answer, pipeline_info)
        trace.cost = {"tokens": tokens, "cost_yuan": cost, "cache_hit": False}

        # ---- 后置：PII 脱敏 ----
        sanitized_answer, redaction_log = self.pii_sanitizer.sanitize(answer)
        trace.pii_redaction = redaction_log
        trace.final_answer = sanitized_answer

        # ---- 写入缓存（只缓存直答和生成结果，不缓存兜底） ----
        if branch in ("direct_answer", "llm_generate"):
            self._cache[cache_key] = (sanitized_answer, pipeline_info, time.time())

        self._save_trace(trace)
        return sanitized_answer, trace

    def _estimate_cost(self, branch: str, answer: str, pipeline_info: dict) -> tuple:
        """估算本次请求的 token 消耗和费用（元）"""
        # 粗估 token：中文 1 字 ≈ 1 token
        query_tokens = 10  # 平均查询长度
        answer_tokens = len(answer)

        if branch == "direct_answer":
            # 直答：只有 embedding
            total_tokens = query_tokens
            cost = (query_tokens / 1000) * self.PRICING["embedding"]
        elif branch == "llm_generate":
            # 生成：embedding + LLM 输入(检索结果~500字) + LLM 输出
            retrieved_tokens = sum(len(d.get("content", "")) for d in pipeline_info.get("retrieved_docs", []))
            llm_input = query_tokens + retrieved_tokens + 100  # +prompt
            llm_output = answer_tokens
            total_tokens = query_tokens + llm_input + llm_output
            cost = (query_tokens / 1000) * self.PRICING["embedding"] + \
                   (llm_input / 1000) * self.PRICING["llm_input"] + \
                   (llm_output / 1000) * self.PRICING["llm_output"]
            if pipeline_info.get("rerank_applied"):
                cost += (retrieved_tokens / 1000) * self.PRICING["rerank_llm"]
        else:
            # 兜底：只有 embedding
            total_tokens = query_tokens
            cost = (query_tokens / 1000) * self.PRICING["embedding"]

        return total_tokens, round(cost, 6)

    def _save_trace(self, trace: TraceRecord):
        """保存 trace 到内存 + JSONL 文件（持久化审计）"""
        self.trace_store.append(trace.to_dict())
        # 持久化到 JSONL 文件（一行一条，追加写入）
        try:
            log_dir = os.environ.get("TRACE_LOG_DIR", "/Users/feng/Documents/trae_projects/zjj/trace_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "traces.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"⚠️ Trace 持久化失败: {e}")

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """根据 trace_id 查询历史 trace"""
        for t in self.trace_store:
            if t["trace_id"] == trace_id:
                return t
        return None

    def get_all_traces(self) -> list:
        """获取所有 trace"""
        return self.trace_store
