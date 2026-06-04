"""
AST-LLM Unified - Unit Tests
============================

テスト項目:
  - ASTNode のハッシュチェーン検証
  - Scope のシンボルテーブル動作
  - TransformPipeline の冪等性チェック
  - Tokenizer の語彙構築
  - EmbeddingLayer のベクトル化
  - ASTLLMUnified の推論動作
"""

import json
import numpy as np
from ast_llm_unified import (
    ASTNode, Scope, Tokenizer, EmbeddingLayer,
    AttentionHead, VectorMemory, TransformPipeline,
    TransformPass, ASTLLMUnified, TransformError
)


# ═══════════════════════════════════════════════════════════
# 1. ASTNode & ハッシュチェーン テスト
# ═══════════════════════════════════════════════════════════

def test_astnode_hash_chain():
    """ASTNode のハッシュチェーン検証"""
    # 子ノード
    left = ASTNode("Identifier", value="x")
    right = ASTNode("Identifier", value="y")
    
    # 親ノード
    binop = ASTNode("BinaryOp", value="+", children=[left, right])
    
    # ハッシュ計算
    binop._hash = binop.compute_hash()
    binop.node_id = binop._hash
    
    # 検証
    assert binop.verify(), "ハッシュ検証失敗"
    assert len(binop._hash) == 16, "ハッシュ長が不正"
    print("✓ ASTNode ハッシュチェーン OK")


def test_astnode_walk():
    """ASTNode 深さ優先走査"""
    # ツリー構築
    leaf1 = ASTNode("Literal", value="1")
    leaf2 = ASTNode("Literal", value="2")
    parent = ASTNode("Add", children=[leaf1, leaf2])
    
    # 走査
    nodes = list(parent.walk())
    assert len(nodes) == 3, f"走査ノード数が不正: {len(nodes)}"
    assert nodes[0] == parent, "親ノードが最初"
    print("✓ ASTNode 走査 OK")


# ═══════════════════════════════════════════════════════════
# 2. Scope & シンボルテーブル テスト
# ═══════════════════════════════════════════════════════════

def test_scope_define_lookup():
    """Scope のシンボルテーブル"""
    global_scope = Scope("global")
    local_scope = Scope("function", parent=global_scope)
    
    # グローバルに定義
    global_scope.define("g", {"type": "int"})
    
    # ローカルに定義
    local_scope.define("x", {"type": "str"})
    
    # 検索（ローカルから）
    assert local_scope.lookup("x") == {"type": "str"}
    assert local_scope.lookup("g") == {"type": "int"}  # 親から検索
    assert local_scope.lookup("undefined") is None
    
    print("✓ Scope シンボルテーブル OK")


def test_scope_hash():
    """Scope ハッシュ計算"""
    scope1 = Scope("test")
    scope1.define("a", {})
    scope1.define("b", {})
    
    hash1 = scope1.scope_hash()
    assert isinstance(hash1, str) and len(hash1) == 8
    
    # 順序が異なってもハッシュは同じ（ソート済み）
    scope2 = Scope("test")
    scope2.define("b", {})
    scope2.define("a", {})
    hash2 = scope2.scope_hash()
    
    assert hash1 == hash2, "スコープハッシュが異なる"
    print("✓ Scope ハッシュ OK")


# ═══════════════════════════════════════════════════════════
# 3. Tokenizer テスト
# ═══════════════════════════════════════════════════════════

def test_tokenizer_fit():
    """Tokenizer 語彙構築"""
    tokenizer = Tokenizer()
    
    texts = [
        "hello world",
        "hello python",
        "python is great"
    ]
    
    tokenizer.fit(texts, min_freq=1)
    
    # 特別トークン + 語彙
    assert tokenizer.vocab_size >= 5  # <PAD>, <UNK>, <BOS>, <EOS> + 語彙
    print(f"✓ Tokenizer 語彙構築 OK (vocab_size={tokenizer.vocab_size})")


def test_tokenizer_encode_decode():
    """Tokenizer エンコード/デコード"""
    tokenizer = Tokenizer()
    tokenizer.fit(["hello world hello"])
    
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    assert isinstance(ids, list) and len(ids) > 0
    assert isinstance(decoded, str)
    print(f"✓ Tokenizer エンコード/デコード OK")
    print(f"  入力: {text}")
    print(f"  ID列: {ids}")
    print(f"  復号: {decoded}")


# ═══════════════════════════════════════════════════════════
# 4. EmbeddingLayer テスト
# ═══════════════════════════════════════════════════════════

def test_embedding_forward():
    """EmbeddingLayer 前向き計算"""
    embedding = EmbeddingLayer(vocab_size=100, d_model=64)
    
    token_ids = [1, 2, 3, 4, 5]
    output = embedding.forward(token_ids)
    
    assert output.shape == (5, 64), f"出力shape不正: {output.shape}"
    assert output.dtype == np.float32
    print(f"✓ EmbeddingLayer 前向き計算 OK (shape={output.shape})")


def test_embedding_expand_vocab():
    """EmbeddingLayer 語彙拡張"""
    embedding = EmbeddingLayer(vocab_size=50, d_model=64)
    
    original_size = len(embedding.W)
    embedding.expand_vocab(100)
    
    assert len(embedding.W) == 100, "語彙拡張失敗"
    assert len(embedding.W) > original_size
    print(f"✓ EmbeddingLayer 語彙拡張 OK ({original_size} → {len(embedding.W)})")


# ═══════════════════════════════════════════════════════════
# 5. AttentionHead テスト
# ═══════════════════════════════════════════════════════════

def test_attention_forward():
    """AttentionHead 前向き計算"""
    head = AttentionHead(d_model=64, d_k=32)
    
    X = np.random.randn(5, 64).astype(np.float32)  # [seq_len, d_model]
    output, weights = head.forward(X)
    
    assert output.shape == (5, 32), f"出力shape不正: {output.shape}"
    assert weights.shape == (5, 5), f"重みshape不正: {weights.shape}"
    assert np.allclose(weights.sum(axis=-1), 1.0), "重みの合計が1でない"
    
    print(f"✓ AttentionHead 前向き計算 OK")
    print(f"  出力: {output.shape}, 重み: {weights.shape}")


# ═══════════════════════════════════════════════════════════
# 6. VectorMemory テスト
# ═══════════════════════════════════════════════════════════

def test_vector_memory_add_search():
    """VectorMemory 追加・検索"""
    memory = VectorMemory(d_model=64)
    
    # メモリに追加
    key1 = np.random.randn(64).astype(np.float32)
    val1 = np.random.randn(64).astype(np.float32)
    memory.add("text1", key1, val1, source="doc")
    
    key2 = np.random.randn(64).astype(np.float32)
    val2 = np.random.randn(64).astype(np.float32)
    memory.add("text2", key2, val2, source="doc")
    
    # 検索
    query = np.random.randn(64).astype(np.float32)
    results = memory.search(query, top_k=2)
    
    assert len(results) == 2, "検索結果数が不正"
    assert all(isinstance(text, str) for text, _ in results)
    assert all(isinstance(score, float) for _, score in results)
    
    print(f"✓ VectorMemory 検索 OK ({memory.size} 件)")


# ═══════════════════════════════════════════════════════════
# 7. TransformPipeline テスト
# ═══════════════════════════════════════════════════════════

def test_transform_pipeline_idempotent():
    """TransformPipeline 冪等性チェック"""
    pipeline = TransformPipeline()
    
    # 決定的な変換関数
    def deterministic_transform(node: ASTNode, scope: Scope) -> str:
        return json.dumps({"type": node.node_type, "value": node.value})
    
    def validate_json(node: ASTNode, output: str) -> list[str]:
        try:
            json.loads(output)
            return []
        except:
            return ["JSON解析失敗"]
    
    pipeline.add_pass(TransformPass(
        name="json_transform",
        fn=deterministic_transform,
        validator=validate_json
    ))
    
    # 実行
    node = ASTNode("Test", value="hello")
    scope = Scope("global")
    
    result = pipeline.run(node, scope)
    assert isinstance(result, str)
    print("✓ TransformPipeline 冪等性チェック OK")


def test_transform_pipeline_non_idempotent():
    """TransformPipeline 非冪等性検出"""
    import random
    
    pipeline = TransformPipeline()
    
    # 非決定的な変換関数（エラーが出るはず）
    def non_deterministic_transform(node: ASTNode, scope: Scope) -> str:
        return str(random.random())  # 毎回異なる値
    
    pipeline.add_pass(TransformPass(
        name="random_transform",
        fn=non_deterministic_transform
    ))
    
    node = ASTNode("Test")
    scope = Scope("global")
    
    try:
        pipeline.run(node, scope)
        assert False, "冪等性違反を検出できなかった"
    except TransformError as e:
        assert "冪等性違反" in str(e)
        print("✓ TransformPipeline 非冪等性検出 OK")


# ═══════════════════════════════════════════════════════════
# 8. ASTLLMUnified 統合テスト
# ═══════════════════════════════════════════════════════════

def test_astllm_build_knowledge():
    """ASTLLMUnified 知識ベース構築"""
    llm = ASTLLMUnified(d_model=32, n_heads=2)
    
    docs = [
        "AST is a tree structure.",
        "LLM uses attention mechanism.",
        "Hashing detects tampering.",
    ]
    
    llm.build_knowledge(docs, verbose=False)
    
    stats = llm.stats()
    assert stats["memory_size"] == 3
    assert stats["vocab_size"] > 0
    print(f"✓ ASTLLMUnified 知識構築 OK (vocab={stats['vocab_size']}, memory={stats['memory_size']})")


def test_astllm_parse_prompt():
    """ASTLLMUnified プロンプト解析"""
    llm = ASTLLMUnified()
    
    # 質問型プロンプト
    ast_q = llm.parse_prompt_ast("What is AST?")
    assert ast_q.node_type == "Prompt"
    assert ast_q.value in ["question", "statement"]
    
    # 陳述型プロンプト
    ast_s = llm.parse_prompt_ast("AST is important.")
    assert ast_s.node_type == "Prompt"
    
    print(f"✓ ASTLLMUnified プロンプト解析 OK")


def test_astllm_generate():
    """ASTLLMUnified 推論"""
    llm = ASTLLMUnified(d_model=32, n_heads=2, max_tokens=80)
    
    docs = [
        "Python is a programming language created in 1991.",
        "Java was originally called Oak.",
        "C++ is an extension of C language.",
    ]
    
    llm.build_knowledge(docs, verbose=False)
    
    answer = llm.generate("Tell me about Python", top_k=2, verbose=False)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "。" in answer or "." in answer or answer.strip()  # 句点が付くか英文
    
    print(f"✓ ASTLLMUnified 推論 OK")
    print(f"  Q: Tell me about Python")
    print(f"  A: {answer[:60]}...")


def test_astllm_add_knowledge():
    """ASTLLMUnified オンライン知識追加"""
    llm = ASTLLMUnified(d_model=32, n_heads=2)
    
    llm.build_knowledge(["Initial knowledge"], verbose=False)
    initial_size = llm.memory.size
    
    llm.add_knowledge("New knowledge added online", source="user")
    
    assert llm.memory.size > initial_size
    print(f"✓ ASTLLMUnified 知識追加 OK ({initial_size} → {llm.memory.size})")


# ═══════════════════════════════════════════════════════════
# テスト実行
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("  AST-LLM Unified - Unit Tests")
    print("=" * 62)
    
    # グループ1: ASTNode & ハッシュ
    print("\n[1] ASTNode & ハッシュチェーン")
    test_astnode_hash_chain()
    test_astnode_walk()
    
    # グループ2: Scope
    print("\n[2] Scope & シンボルテーブル")
    test_scope_define_lookup()
    test_scope_hash()
    
    # グループ3: Tokenizer
    print("\n[3] Tokenizer")
    test_tokenizer_fit()
    test_tokenizer_encode_decode()
    
    # グループ4: EmbeddingLayer
    print("\n[4] EmbeddingLayer")
    test_embedding_forward()
    test_embedding_expand_vocab()
    
    # グループ5: AttentionHead
    print("\n[5] AttentionHead")
    test_attention_forward()
    
    # グループ6: VectorMemory
    print("\n[6] VectorMemory")
    test_vector_memory_add_search()
    
    # グループ7: TransformPipeline
    print("\n[7] TransformPipeline")
    test_transform_pipeline_idempotent()
    test_transform_pipeline_non_idempotent()
    
    # グループ8: ASTLLMUnified 統合
    print("\n[8] ASTLLMUnified (統合)")
    test_astllm_build_knowledge()
    test_astllm_parse_prompt()
    test_astllm_generate()
    test_astllm_add_knowledge()
    
    print("\n" + "=" * 62)
    print("  全テスト成功 ✅")
    print("=" * 62)
