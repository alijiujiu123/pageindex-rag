"""Issue #1: Import verification and config loading tests."""


def test_import_pageindex_rag():
    import pageindex_rag
    assert pageindex_rag.__version__ == "0.1.0"


def test_import_submodules():
    import pageindex_rag.storage
    import pageindex_rag.retrieval
    import pageindex_rag.search
    import pageindex_rag.pipeline
    import pageindex_rag.api
    import pageindex_rag.benchmark
    import pageindex_rag.ingestion


def test_config_defaults(config):
    assert config.model == "gpt-4o-2024-11-20"
    assert config.openai_api_key == "test-key"
    assert "sqlite" in config.database_url


def test_config_override():
    from pageindex_rag.config import get_config
    cfg = get_config(model="gpt-4o-mini", semantic_top_k=50)
    assert cfg.model == "gpt-4o-mini"
    assert cfg.semantic_top_k == 50


def test_pageindex_source_available():
    from pageindex.utils import extract_json, get_nodes, structure_to_list
    assert callable(extract_json)
    assert callable(get_nodes)
    assert callable(structure_to_list)
