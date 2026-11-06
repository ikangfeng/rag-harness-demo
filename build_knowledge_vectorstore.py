"""知识库构建代码"""
#%%
#导入QA型知识库
import pandas as pd
from config import rag_test_Config

# 延迟加载数据，只在需要时读取
qa_raw_data = None
doc_raw_data = None

def load_data():
    """延迟加载数据，只在需要时读取"""
    global qa_raw_data, doc_raw_data
    if qa_raw_data is None:
        qa_raw_data = pd.read_csv("/Users/feng/Desktop/大模型 课件/代码/rag_lecture_materials/金融QA数据集1.csv")
    if doc_raw_data is None:
        doc_raw_data = pd.read_csv("/Users/feng/Desktop/大模型 课件/代码/rag_lecture_materials/文档型知识.txt", header=None, names=["content"])
    return qa_raw_data, doc_raw_data
#%%
"""
创建统一知识库,会有三类知识：
文档型知识->文档分块知识
QA型知识:->(1)基于query高精度匹配的知识 (2)基于qa中精度匹配的知识
需要以统一的格式写入同一个知识库,然后通过meta_datas中指定不同类型的type进行区分
"""
from typing import List, Optional, Dict, Any
import uuid
import chromadb
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#%%
    #1.创建query型知识在向量库中的数据结构,需要将answer放入metadatas中
def init_vectordb(qa_raw_data=None,api_key=rag_test_Config().api_key):
    # 如果没有提供数据，使用延迟加载的数据
    if qa_raw_data is None:
        qa_raw_data, _ = load_data()
    
    query_list_document_list=[]
    raw_query_list_document_list = [Document(page_content=q) for q in qa_raw_data["Question"]]
    for i,doc in enumerate(raw_query_list_document_list):
        doc.metadata["type"]="query"
        doc.metadata["answer"]=qa_raw_data["Answer"][i]
        doc.metadata["id"]=str(uuid.uuid4())
        query_list_document_list.append(doc)

    #2.创建qa型知识在向量库中的数据结构，需要将问题和答案拼接,通过答案：进行标识
    qa_list_document_list=[]
    raw_qa_list_document_list = [Document(page_content=q+"答案："+a) for q,a in zip(qa_raw_data["Question"], qa_raw_data["Answer"])]
    for i,doc in enumerate(raw_qa_list_document_list):
        doc.metadata["type"]="qa"
        doc.metadata["id"]=str(uuid.uuid4())
        qa_list_document_list.append(doc)

    #3.长文档建立向量库的元数据
    loader = TextLoader("/Users/feng/Desktop/大模型 课件/代码/rag_lecture_materials/文档型知识.txt")
    doc_documents = loader.load_and_split()
    # 2. 文本分割（按段落/句子切分）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,      # 每个分块的最大字符数，设置的小一些这里
        chunk_overlap=5,   # 分块之间的重叠字符数
        separators=["\n\n", "\n", "。", "？", "！", "；"]  # 中文分割符
    )
    #进行文档分割
    splits_documents = text_splitter.split_documents(doc_documents)
    #对documents的元数据做一下修改增加type类型
    doc_list_document_list = []
    for i,doc in enumerate(splits_documents):
        doc.metadata["type"]="docs"
        doc.metadata["id"]=str(uuid.uuid4())
        doc_list_document_list.append(doc)
    #将三类知识合并成在一起,这个combine_document_list就是后续构建向量库用的统一数据结构了
    combine_document_list=query_list_document_list
    combine_document_list.extend(qa_list_document_list)
    combine_document_list.extend(doc_list_document_list)

    #构建向量库
    #载入向量模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v2",dashscope_api_key=api_key)
    
    # 尝试从持久化目录加载向量库
    import os
    persist_directory = os.environ.get("RAG_VECTORSTORE_DIR", "./vectorstore")
    
    # 如果持久化目录存在，直接加载向量库
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            collection_name="novel",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    
    # 否则创建新的向量库
    # 去重处理，基于page_content字段
    unique_documents = []
    seen_page_contents = set()
    for doc in combine_document_list:
        if doc.page_content not in seen_page_contents:
            seen_page_contents.add(doc.page_content)
            unique_documents.append(doc)
    
    # 创建新的向量库，添加持久化目录
    vectorstore = Chroma.from_documents(
        unique_documents, 
        embeddings, 
        collection_name="novel",
        persist_directory=persist_directory  # 添加持久化目录
    )
    # 然后你就可以像平常一样使用vectorstore了
    return vectorstore
#%%
# 移除直接调用，避免重复初始化
# # vectorstore=init_vectordb(qa_raw_data)
"""
构建三个不同score_threshold(检索精度)的检索器:
high_resolution_retriever(高精度检索器):score_threshold=0.82 用于匹配精确命中type=query 类的知识
middle_resolution_retriever(中等精度检索器):score_threshold=0.68 用于匹配与type=qa相关的所有内容
doc_retriver(文档型检索器)：用于匹配type=docs相关的内容，score_threshold=0.65作为前两类检索器都没有检索到结果时候的补充检索
"""
# 以下检索器构建代码已移至chat_help_main.py中
# high_resolution_retriever = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.82},
# )
# # query="解释Black-Scholes期权定价模型的核心假设"
# # docs = high_resolution_retriever.get_relevant_documents(query, filter={"type": "query"}) #通过添加filer来过滤查找范围为query标签的内容
# # print("docs:",docs)
# #%%
# middle_resolution_retriever=vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.68},
# )
# # query="解释Black-Scholes期权定价模型的核心假设"
# # docs = middle_resolution_retriever.get_relevant_documents(query, filter={"type": "qa"}) #通过添加filer来过滤查找范围为qa标签的内容
# # print("docs:",docs)
# #%%
# doc_retriever=vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.3},
# )
# 注释掉所有使用vectorstore的测试代码
# # query="n数据悖论：Q2 GDP增速2.4%但贫富分化加剧基尼系数0.49是什么原因"
# # docs = doc_retriever.get_relevant_documents(query, filter={"type": "docs"}) #通过添加filer来过滤查找范围为qa标签的内容
# # print("docs:",docs)
# #%%
# # query="n数据悖论：Q2 GDP增速2.4%但贫富分化加剧（基尼系数0.49）"
# # docs = doc_retriever.get_relevant_documents(query, filter={"type": "docs"}) #通过添加filer来过滤查找范围为qa标签的内容
# # print("docs:",docs)
# # print([k.page_content for k in docs])
# #%%
#使用bm25retriever进行检索
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import jieba
def chinese_tokenizer(text: str):
    tokens = jieba.lcut(text)
    return [token for token in tokens] # if token not in stopwords.words('chinese')

#%%
#---------- (5) 构建bm25 retriever器 -------------
#对于bm25类的retriever我们将qa类型的数据和文档类型的数据进行拼接同步进行检索
def get_bm25_retriever(K): #默认分词器基于英文空格作为拆分符号进行分词的
    # 使用延迟加载的数据
    qa_raw_data, _ = load_data()
    
    bm25_doc_list = []
    # 追加上QA类型知识
    bm25_doc_list.extend([q + "答案：" + a for q, a in zip(qa_raw_data["Question"], qa_raw_data["Answer"])])
    # 追加上文档类型知识
    loader = TextLoader("/Users/feng/Desktop/大模型 课件/代码/rag_lecture_materials/文档型知识.txt")
    doc_documents = loader.load_and_split()
    # 2. 文本分割（按段落/句子切分）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,  # 每个分块的最大字符数，设置的小一些这里
        chunk_overlap=5,  # 分块之间的重叠字符数
        separators=["\n\n", "\n", "。", "？", "！", "；"]  # 中文分割符
    )
    # 进行文档分割
    splits_documents = text_splitter.split_documents(doc_documents)
    doc_bm25_list = [k.page_content for k in splits_documents]
    bm25_doc_list.extend(doc_bm25_list)
    # 使用示例
    bm25_retriever = BM25Retriever.from_texts( #Create a BM25Retriever from a list of texts.
        bm25_doc_list, metadatas=[{"type": "bm25"}] * len(bm25_doc_list),
        preprocess_func=chinese_tokenizer, #自定义分词器
    )
    bm25_retriever.k = K #设置检索条目数=2
    #使用Bm25检索器对文本进行检索
    return bm25_retriever
#%%

#根据提供的背景检索材料以及LLM自由生成答案
from langchain_community.llms import Tongyi  # 通义千问的LangChain集成
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
# 设置DashScope API Key（需在阿里云控制台获取）就是api_key
import dashscope
import sys
sys.path.append("./project2_chat_help_system")
from config import rag_test_Config
import os
os.environ["DASHSCOPE_API_KEY"] = rag_test_Config().api_key
# 初始化Qwen-Max模型
llm = Tongyi(
    model_name="qwen-max",  # 使用qwen-max模型
    temperature=0.7,        # 控制随机性 (0~1)
    top_p=0.9              # 控制多样性 (0~1)
)

# 全局创建PromptTemplate和LLMChain，避免每次调用都重新创建
prompt_template = """你是一个专业的金融知识问答助手，**仅回答金融领域的问题**。基于背景信息回答用户的问题。如果背景信息与用户的问题相关则需要结合背景信息进行回答(不能乱编)，
    ✅ **可以回答的问题**：
- 金融产品知识（股票、债券、基金、期货、期权等）
- 投资理财概念（市盈率、市净率、资产配置等）
- 金融市场分析（宏观经济、货币政策、行业分析等）
- 风险管理（对冲、止损、分散投资等）

    ❌ **严格禁止**：
1. 预测股票/指数涨跌
2. 推荐具体投资标的
3. 提供个性化投资建议
4. **回答非金融领域的问题**（如娱乐、天气、生活、编程等）
🔍 **判断逻辑**：
- **如果问题不属于金融领域 → 使用以下标准回复**
【非金融问题标准回复】
"抱歉，我是专注于金融知识的问答助手，主要回答金融产品、投资理财、市场分析等相关问题。您的问题超出了我的专业范围，建议您咨询相关领域的专家。"
- 如果问题属于金融领域 + 背景知识足够 → 详细回答并引用背景知识
- 如果问题属于金融领域 + 背景知识不足 → 告知"暂无相关信息"并建议查询渠道
- 如果问题属于预测/推荐类 → 明确拒绝并提示风险

【输出要求】直接输出答案正文，不要有任何问候语或自我介绍。

    🔒 安全约束：如果背景信息与用户问题无关或背景信息中没有足够支撑，请直接回答"暂无相关信息"，禁止基于用户问题关键词自行编造内容。
    背景信息:{background_knowledge}
    历史记录:{history_message}
    用户问题:{current_query}
    回答："""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["background_knowledge", "history_message", "current_query"]
)
# 创建LLM Chain
chain = LLMChain(llm=llm, prompt=prompt)

#调用LLMChain获取对应的结果
def generation_replay_by_LLM(background_knowledge, history_message, current_query, llm=llm):
    """
    :param background_knowledge: 通过检索得到的背景信息字符串
    :param history_message: 上下文历史记录，这里会组合成一个字符串
    :param current_query: 当前用户问题
    :return:
    """
    # 调用模型并返回结果
    response = chain.run({
        "current_query": current_query, 
        "history_message": history_message,
        "background_knowledge": background_knowledge
    })
    return response

#流式调用LLMChain获取对应的结果
def generation_replay_by_LLM_stream(background_knowledge, history_message, current_query, llm=llm):
    """
    流式版本的LLM调用函数，实时返回生成的token
    :param background_knowledge: 通过检索得到的背景信息字符串
    :param history_message: 上下文历史记录，这里会组合成一个字符串
    :param current_query: 当前用户问题
    :yield: 每个生成的token（字符串片段）
    """
    # 格式化prompt
    formatted_prompt = prompt.format(
        background_knowledge=background_knowledge,
        history_message=history_message,
        current_query=current_query
    )
    
    # 使用流式调用，实时yield每个token
    for chunk in llm.stream(formatted_prompt):
        yield chunk

# background_knowledge=("美联储维持5.25%-5.5%利率区间已达12个月，欧元区跟进紧缩，而日本央行结束负利率时代却步履蹒跚。" #知识1
#                      "市场对降息预期的反复修正引发美债收益率震荡（10年期收益率4.2%→3.8%→4.1%）") #知识2
# history_message="""用户:你好\nAI:你好，有什么能帮到您的？"""
# current_query="美联储维持的利率引发欧元和美债的影响是什么？"
# response=generation_replay_by_LLM(background_knowledge,history_message,current_query)
# print("response:",response)