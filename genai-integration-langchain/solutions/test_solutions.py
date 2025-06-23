def test_simple_agent(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "simple_agent"
    )
    
    assert output.startswith("Answer:")

def test_neo4j_query(test_helpers, monkeypatch):
    
    output = test_helpers.run_module(
        monkeypatch, 
        "neo4j_query"
    )
    
    assert output > ""

def test_schema_agent(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "schema_agent"
    )
    
    assert output.startswith("Answer:")

def test_vector_search(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_search"
    )
    
    assert output > ""

def test_vector_retriever(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_retriever"
    )
    
    assert output.startswith("Answer:")

def test_vector_graph_retriever(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_graph_retriever"
    )
    
    assert output.startswith("Answer:")

def test_cypher_qa(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "cypher_qa_prompt"
    )
    
    assert output > ""

def test_cypher_retriever(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "cypher_retriever_enhanced"
    )
    
    assert output > ""