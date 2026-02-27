# CLAUDE.md

This file provides guidance to Claude Code when working with the `pageindex-rag` repository.

## Project Overview

`pageindex-rag` 是基于 [PageIndex](https://github.com/alijiujiu123/PageIndex) 的 reasoning-based RAG 系统。核心思路：将 PDF/MD 转换为层级树结构（类似 TOC），用 LLM 推理定位相关节点，而非向量检索分块。目标：FinanceBench 基准测试 98.7% 准确率。

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env  # 填入 API Key
```

`.env` 必填项：
```
CHATGPT_API_KEY=your_openai_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pageindex_rag
CHROMA_PERSIST_DIR=./chroma_data
```

## Running Tests

```bash
pytest tests/              # 全量测试
pytest tests/test_init.py  # 单文件测试
pytest -v -k "test_name"   # 指定测试
```

## Architecture

### 目录结构

```
pageindex/              # 直接复制的 PageIndex 源码（不要修改）
pageindex_rag/
├── config.py           # 环境变量配置，返回 SimpleNamespace
├── storage/            # PostgreSQL 文档存储
│   ├── models.py       # SQLAlchemy 模型（documents 表）
│   └── document_store.py  # DocumentStore CRUD 类
├── retrieval/          # 检索层
│   ├── tree_search.py  # LLM 推理式树搜索
│   └── node_extractor.py  # 节点内容提取（PDF → text）
├── search/             # 文档搜索层
│   ├── description_search.py  # 描述匹配搜索
│   ├── metadata_search.py     # LLM → SQL 元数据搜索
│   ├── semantic_search.py     # ChromaDB 语义搜索
│   ├── embeddings.py          # embedding 工具
│   └── router.py              # 搜索策略路由器
├── ingestion/          # 文档入库
│   └── ingest.py       # PDF/MD → 树结构 → 存储 → 建索引
├── pipeline/           # RAG 编排
│   ├── rag_pipeline.py     # 主 Pipeline（单/多文档模式）
│   └── answer_generator.py # 答案生成（含页码引用）
├── api/                # FastAPI 服务
│   ├── app.py
│   └── routes/
│       ├── documents.py  # 文档管理端点
│       ├── search.py     # 搜索端点
│       └── qa.py         # 问答端点
└── benchmark/          # FinanceBench 评估
    ├── financebench.py  # 数据集加载
    └── evaluator.py     # LLM 等价性判定 + 报告
```

### Issue 依赖顺序

```
#1(done) → #2 → #3 → #6, #7
                #2 → #4, #8 → #9
#1 → #5 → #10(←#4,#9) → #14 → #15 → #16 → #17
#2 → #11(←#8) → #12 → #13(←#10) → #18(←#17)
```

### PageIndex 核心模块（pageindex/ 目录，只读复用）

| 函数/类 | 位置 | 用途 |
|---------|------|------|
| `page_index_main()` | `page_index.py` | PDF → 树结构 JSON |
| `md_to_tree()` | `page_index_md.py` | MD → 树结构 JSON |
| `ChatGPT_API` / `ChatGPT_API_async` | `utils.py` | OpenAI 调用（含 10 次重试） |
| `extract_json()` | `utils.py` | 解析 LLM 返回的 JSON |
| `get_text_of_pdf_pages()` | `utils.py` | 按页码提取 PDF 文本 |
| `get_page_tokens()` | `utils.py` | 提取 PDF 每页文本+token数 |
| `structure_to_list()` | `utils.py` | 树结构展平为列表 |
| `get_nodes()` | `utils.py` | 获取树中所有节点 |
| `generate_doc_description()` | `utils.py` | LLM 生成文档一句话描述 |
| `ConfigLoader` | `utils.py` | 加载 config.yaml 默认配置 |

### 树节点 Schema

```json
{
  "title": "Chapter 1",
  "node_id": "0001",
  "start_index": 1,
  "end_index": 10,
  "summary": "LLM 生成的摘要（可选）",
  "nodes": []
}
```

### doc_id 格式

`pi-<uuid4>`，例如 `pi-550e8400-e29b-41d4-a716-446655440000`

### DocScore 公式（语义搜索文档聚合）

```
DocScore = 1/√(N+1) × Σ ChunkScore(n)
```
N = 该文档被命中的 chunk 总数，ChunkScore 来自 ChromaDB 相似度。

## Key Design Patterns

- `config` 全程使用 `types.SimpleNamespace`，通过 `pageindex_rag/config.py` 的 `get_config()` 获取
- LLM 调用统一复用 `pageindex/utils.py` 中的 `ChatGPT_API_async`（异步）和 `ChatGPT_API`（同步）
- SQL 查询必须使用参数化查询，防止注入
- 所有测试用 mock 替代真实 LLM 和数据库调用
- `pageindex/` 目录的代码不要修改，只复用

## Development Workflow

每个 Issue 的开发流程：
1. 创建 feature 分支：`git checkout -b feature/issue-N-description`
2. 编写测试（TDD）
3. 实现功能直到测试通过
4. `git push` + `gh pr create` 关联对应 Issue
5. PR 合并后关闭 Issue

## Deployment

部署目标：`dongjingTest` (root@43.167.189.165)

技术栈：
- PostgreSQL（文档元数据 + 树结构存储）
- ChromaDB（节点 embedding）
- FastAPI + uvicorn
- Docker + docker-compose
- nginx 反代

## Benchmark

FinanceBench（HuggingFace: `PatronusAI/financebench`）：150 道财报问题，目标准确率 ≥ 98.7%。

测试数据类型：10-K / 10-Q SEC 财报，元数据字段：company, fiscal_year, filing_type。
