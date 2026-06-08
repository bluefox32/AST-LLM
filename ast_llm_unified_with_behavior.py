"""
AST-LLM + 行動モデリング 統合フレームワーク
===========================================

【設計】
  Phase 1: 行動AST化 → Parse + 構造化
  Phase 2: 報酬計算 → Transform + 最適化
  Phase 3: 予測 → RandomForest + Attention
  Phase 4: 評価 → LLM + Answer AST
  Phase 5: フィードバック → 冪等性チェック + 再訓練

【特徴】
  - Sinbionations (行動モデリング) を AST構造で統合
  - パラメータ吸収を制御可能に
  - iOS/Siri 対応メタデータ
  - インクリメンタル学習対応
"""

from __future__ import annotations

import os
import re
import math
import hashlib
import json
import time
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ═══════════════════════════════════════════════════════════
# 1. ASTノード定義（行動対応）
# ═══════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """汎用ASTノード（行動構造化対応）"""
    node_type: str
    value: Optional[str] = None
    children: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    
    node_id: Optional[str] = field(default=None, repr=False)
    parent_hash: Optional[str] = field(default=None, repr=False)
    _hash: Optional[str] = field(default=None, repr=False)

    def compute_hash(self, parent_hash: Optional[str] = None) -> str:
        """ハッシュチェーン計算"""
        payload = json.dumps({
            "type": self.node_type,
            "value": self.value,
            "meta": self.meta,
            "children": [c.compute_hash() for c in self.children],
            "parent": parent_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def verify(self) -> bool:
        """ハッシュ検証"""
        return self._hash == self.compute_hash(self.parent_hash)

    def walk(self):
        """深さ優先走査"""
        yield self
        for child in self.children:
            yield from child.walk()


# ═══════════════════════════════════════════════════════════
# 2. スコープ付きシンボルテーブル
# ═══════════════════════════════════════════════════════════

class Scope:
    """スコープ＆シンボルテーブル"""
    
    def __init__(self, name: str, parent: Optional[Scope] = None):
        self.name = name
        self.parent = parent
        self._table: dict[str, dict] = {}

    def define(self, symbol: str, info: dict):
        self._table[symbol] = info

    def lookup(self, symbol: str) -> Optional[dict]:
        if symbol in self._table:
            return self._table[symbol]
        if self.parent:
            return self.parent.lookup(symbol)
        return None

    def scope_hash(self) -> str:
        keys = sorted(self._table.keys())
        return hashlib.md5(json.dumps(keys).encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════
# 3. 行動モデル（Sinbionations統合）
# ═══════════════════════════════════════════════════════════

class BehaviorModel:
    """行動モデリング（Sinbionations組み込み）"""
    
    def __init__(self, initial_data: dict = None):
        """
        initial_data: {
            'statistical': {...},  # 年齢、性別、地域など
            'personalized': {...}  # アクション、環境、成果など
        }
        """
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        
        # 報酬の重み
        self.weights = {
            'work_success': 0.4,
            'financial_gain': 0.3,
            'social_interaction': 0.2,
            'stress_level': -0.1
        }
        
        # 初期データ
        if initial_data:
            self.build_from_data(initial_data)
    
    def build_from_data(self, data: dict):
        """データから行動モデル構築"""
        df_stat = pd.DataFrame(data.get('statistical', {}))
        df_pers = pd.DataFrame(data.get('personalized', {}))
        
        self.df = pd.concat([df_stat, df_pers], axis=1)
        self._compute_rewards()
        self._train_model()
    
    def _compute_rewards(self):
        """報酬計算"""
        self.df['total_reward'] = (
            self.df.get('work_success', 0) * self.weights['work_success'] +
            self.df.get('financial_gain', 0) * self.weights['financial_gain'] +
            self.df.get('social_interaction', 0) * self.weights['social_interaction'] +
            self.df.get('stress_level', 0) * self.weights['stress_level']
        )
    
    def _train_model(self):
        """RandomForest 訓練"""
        feature_cols = ['age', 'gender', 'region', 'economic_status', 
                       'action_id', 'environment_data']
        
        X = self.df[feature_cols]
        y = self.df['total_reward']
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_reward(self, features: dict) -> float:
        """行動の報酬を予測"""
        feature_cols = ['age', 'gender', 'region', 'economic_status', 
                       'action_id', 'environment_data']
        
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        return float(self.model.predict(X_scaled)[0])
    
    def add_feedback(self, action: dict):
        """フィードバック（新しいアクション結果）を追加"""
        # pd.concat で append 代替
        new_row = pd.DataFrame([action])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self._compute_rewards()
        self._train_model()
    
    def get_best_action(self, candidates: list[dict]) -> dict:
        """複数候補から最適行動を選択"""
        best_action = None
        best_reward = -float('inf')
        
        for action in candidates:
            reward = self.predict_reward(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return {
            'action': best_action,
            'reward': best_reward,
            'confidence': min(0.99, 0.8 + (best_reward / 1000))  # 簡易信頼度
        }


# ═══════════════════════════════════════════════════════════
# 4. Tokenizer
# ═══════════════════════════════════════════════════════════

class Tokenizer:
    """単語→ID変換"""
    
    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self):
        self.word2id: dict[str, int] = dict(self.SPECIAL)
        self.id2word: dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
        self._next_id = len(self.SPECIAL)

    def fit(self, texts: list[str], min_freq: int = 1):
        """語彙構築"""
        freq: dict[str, int] = defaultdict(int)
        for text in texts:
            for w in self._split(text):
                freq[w] += 1
        for w, f in sorted(freq.items(), key=lambda x: -x[1]):
            if f >= min_freq and w not in self.word2id:
                self.word2id[w] = self._next_id
                self.id2word[self._next_id] = w
                self._next_id += 1
        return self

    def encode(self, text: str) -> list[int]:
        """テキスト → トークンID列"""
        return [self.word2id.get(w, self.SPECIAL["<UNK>"])
                for w in self._split(text)]

    def _split(self, text: str) -> list[str]:
        """分かち書き"""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+|[^\x00-\x7f]+', text)
        return [tok for tok in tokens if tok.strip()]

    @property
    def vocab_size(self) -> int:
        return self._next_id


# ═══════════════════════════════════════════════════════════
# 5. Embedding層
# ═══════════════════════════════════════════════════════════

class EmbeddingLayer:
    """外積行列 W_E: vocab_size × d_model"""
    
    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (vocab_size + d_model))
        self.W = rng.normal(0, scale, (vocab_size, d_model)).astype(np.float32)
        self.d_model = d_model

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """トークンID列 → [seq_len, d_model]"""
        ids = np.array(token_ids, dtype=np.int32)
        ids = np.clip(ids, 0, len(self.W) - 1)
        emb = self.W[ids]
        emb = emb + self._positional_encoding(len(token_ids))
        return emb

    def _positional_encoding(self, seq_len: int) -> np.ndarray:
        """sin/cos位置エンコーディング"""
        pe = np.zeros((seq_len, self.d_model), dtype=np.float32)
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, self.d_model, 2) *
                     -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:self.d_model // 2])
        return pe

    def expand_vocab(self, new_size: int, seed: int = 0):
        if new_size <= len(self.W):
            return
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (new_size + self.d_model))
        extra = rng.normal(0, scale,
                          (new_size - len(self.W), self.d_model)).astype(np.float32)
        self.W = np.vstack([self.W, extra])


# ═══════════════════════════════════════════════════════════
# 6. Attention Head
# ═══════════════════════════════════════════════════════════

class AttentionHead:
    """Single-head Attention"""
    
    def __init__(self, d_model: int, d_k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = math.sqrt(2.0 / (d_model + d_k))
        self.W_Q = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.W_K = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.W_V = rng.normal(0, s, (d_model, d_k)).astype(np.float32)
        self.d_k = d_k

    def forward(self, X: np.ndarray,
                memory: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """X: [seq_len, d_model], memory: [mem_len, d_model]"""
        src = memory if memory is not None else X
        
        Q = X @ self.W_Q
        K = src @ self.W_K
        V = src @ self.W_V
        
        scores = Q @ K.T / math.sqrt(self.d_k)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        
        output = weights @ V
        return output, weights


# ═══════════════════════════════════════════════════════════
# 7. ベクトルメモリ
# ═══════════════════════════════════════════════════════════

class VectorMemory:
    """インメモリベクトル空間"""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.texts: list[str] = []
        self.sources: list[str] = []

    def add(self, text: str, key_vec: np.ndarray,
            val_vec: np.ndarray, source: str = ""):
        self.keys.append(key_vec.astype(np.float32))
        self.values.append(val_vec.astype(np.float32))
        self.texts.append(text)
        self.sources.append(source)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """内積ベースの検索"""
        if not self.keys:
            return []
        
        K = np.array(self.keys)
        scores = (query_vec @ K.T) / (np.linalg.norm(query_vec) * 
                                       np.linalg.norm(K, axis=1) + 1e-8)
        indices = np.argsort(-scores)[:top_k]
        return [(self.texts[i], float(scores[i])) for i in indices]

    @property
    def size(self) -> int:
        return len(self.texts)


# ═══════════════════════════════════════════════════════════
# 8. Transform パイプライン
# ═══════════════════════════════════════════════════════════

class TransformError(Exception):
    pass


@dataclass
class TransformPass:
    name: str
    fn: Callable[[ASTNode, Scope], str]
    validator: Optional[Callable[[ASTNode, str], list[str]]] = None


class TransformPipeline:
    """AST変換パイプライン"""
    
    def __init__(self):
        self.passes: list[TransformPass] = []

    def add_pass(self, pass_: TransformPass):
        self.passes.append(pass_)

    def run(self, node: ASTNode, scope: Scope) -> str:
        result = ""
        for p in self.passes:
            result = p.fn(node, scope)
            
            if p.validator:
                errors = p.validator(node, result)
                if errors:
                    raise TransformError(f"[{p.name}] 不変条件違反:\n" + 
                                       "\n".join(errors))
            
            result2 = p.fn(node, scope)
            if result != result2:
                raise TransformError(f"[{p.name}] 冪等性違反")
        
        return result


# ═══════════════════════════════════════════════════════════
# 9. 統合LLM + 行動モデル
# ═══════════════════════════════════════════════════════════

class ASTLLMUnifiedWithBehavior:
    """AST-LLM + 行動モデリング 統合版"""
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, max_tokens: int = 120):
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        
        self.tokenizer = Tokenizer()
        self.embedding = EmbeddingLayer(1000, d_model)
        self.heads = [AttentionHead(d_model, d_model // n_heads, seed=i)
                     for i in range(n_heads)]
        self.memory = VectorMemory(d_model)
        
        self.global_scope = Scope("global")
        self.pipeline = TransformPipeline()
        
        # 行動モデル統合
        self.behavior_model = BehaviorModel()

    def build_knowledge(self, documents: list[str], verbose: bool = True):
        """知識ベース構築"""
        self.tokenizer.fit(documents, min_freq=1)
        self.embedding.expand_vocab(self.tokenizer.vocab_size)
        
        for text in documents:
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            
            emb = self.embedding.forward(ids)
            key_vec = emb.mean(axis=0)
            val_vec = emb.max(axis=0)
            self.memory.add(text, key_vec, val_vec, source="doc")
        
        if verbose:
            print(f"[BUILD] {len(documents)} 文書を処理")
            print(f"  語彙: {self.tokenizer.vocab_size}, メモリ: {self.memory.size}")

    def build_behavior_model(self, behavioral_data: dict, verbose: bool = True):
        """行動モデル構築（Sinbionations統合）"""
        self.behavior_model.build_from_data(behavioral_data)
        if verbose:
            print(f"[BEHAVIOR] 行動モデル構築完了")
            print(f"  データ行数: {len(self.behavior_model.df)}")

    def parse_behavior_ast(self, action: dict) -> ASTNode:
        """行動 → AST化"""
        children = [
            ASTNode("Feature", value=f"{k}={v}", meta={"feature_type": k, "value": v})
            for k, v in action.items()
        ]
        
        root = ASTNode("Action", value="behavior",
                      children=children,
                      meta={"action_id": action.get("action_id", 0)})
        root._hash = root.compute_hash()
        root.node_id = root._hash
        
        return root

    def transform_behavior_ast(self, node: ASTNode) -> str:
        """行動AST → 変換"""
        def normalize_pass(n: ASTNode, scope: Scope) -> str:
            return json.dumps({
                "type": n.node_type,
                "features": {c.meta.get("feature_type"): c.meta.get("value")
                            for c in n.children}
            })
        
        self.pipeline.add_pass(TransformPass(
            name="behavior_normalize",
            fn=normalize_pass,
            validator=lambda n, o: [] if o else ["空出力"]
        ))
        
        return self.pipeline.run(node, self.global_scope)

    def evaluate_action(self, action: dict, verbose: bool = False) -> dict:
        """行動を評価（報酬予測 + LLM評価）"""
        
        # Phase 1: 行動をAST化
        action_ast = self.parse_behavior_ast(action)
        if verbose:
            print(f"[ACTION AST] {action_ast}")
        
        # Phase 2: Transform処理
        transformed = self.transform_behavior_ast(action_ast)
        if verbose:
            print(f"[ACTION TRANSFORM] {transformed}")
        
        # Phase 3: 報酬予測
        predicted_reward = self.behavior_model.predict_reward(action)
        if verbose:
            print(f"[PREDICTED REWARD] {predicted_reward:.4f}")
        
        # Phase 4: メタデータ生成（Siri用）
        evaluation_result = {
            "action": action,
            "predicted_reward": float(predicted_reward),
            "node_id": action_ast.node_id,
            "hash": action_ast._hash,
            "confidence": min(0.99, 0.7 + (predicted_reward / 500)),
            "transformed": json.loads(transformed),
            "siri_context": {
                "action_id": action.get("action_id"),
                "reward": float(predicted_reward),
                "confidence": min(0.99, 0.7 + (predicted_reward / 500))
            }
        }
        
        return evaluation_result

    def find_best_action(self, candidates: list[dict], verbose: bool = False) -> dict:
        """複数候補から最適行動を選択"""
        results = []
        
        for action in candidates:
            result = self.evaluate_action(action, verbose=verbose)
            results.append(result)
        
        best = max(results, key=lambda x: x["predicted_reward"])
        
        return {
            "best_action": best,
            "alternatives": results,
            "siri_context": best["siri_context"]
        }

    def add_feedback(self, action_result: dict, verbose: bool = True):
        """フィードバック追加（行動結果を学習）"""
        self.behavior_model.add_feedback(action_result)
        if verbose:
            print(f"[FEEDBACK] 行動結果を統合しました")
            print(f"  新しいデータ行数: {len(self.behavior_model.df)}")

    def stats(self) -> dict:
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "memory_size": self.memory.size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "behavior_model_size": len(self.behavior_model.df) if self.behavior_model.df is not None else 0,
        }


# ═══════════════════════════════════════════════════════════
# 10. デモ実行
# ═══════════════════════════════════════════════════════════

def demo():
    sep = "=" * 70
    print(sep)
    print("  AST-LLM + 行動モデリング 統合フレームワーク")
    print(sep)

    # LLM初期化
    llm = ASTLLMUnifiedWithBehavior(d_model=32, n_heads=2, max_tokens=100)

    # 知識ベース
    knowledge = [
        "高い仕事の成功と金銭獲得は良好な行動を示唆する。",
        "社会交流とストレス管理は生活の質を向上させる。",
        "都市環境での経済的成功は異なるパターンを持つ。",
    ]
    llm.build_knowledge(knowledge, verbose=False)

    # 行動データ
    behavioral_data = {
        'statistical': {
            'age': [25, 35, 45, 55, 65],
            'gender': [0, 1, 0, 1, 0],
            'region': [1, 2, 1, 2, 1],
            'economic_status': [1, 2, 1, 2, 1]
        },
        'personalized': {
            'action_id': [1, 2, 3, 4, 5],
            'environment_data': [10, 20, 30, 40, 50],
            'work_success': [0.8, 0.4, 0.9, 0.2, 1.0],
            'financial_gain': [100, 50, 150, 30, 200],
            'social_interaction': [0.5, 0.3, 0.7, 0.2, 0.9],
            'stress_level': [0.1, 0.2, 0.05, 0.25, 0.0]
        }
    }
    
    llm.build_behavior_model(behavioral_data, verbose=False)

    # 行動評価デモ
    print("\n【行動評価デモ】")
    print("-" * 70)
    
    test_actions = [
        {
            'age': 30, 'gender': 0, 'region': 1, 'economic_status': 1,
            'action_id': 6, 'environment_data': 35,
            'work_success': 0.75, 'financial_gain': 120,
            'social_interaction': 0.65, 'stress_level': 0.12
        },
        {
            'age': 40, 'gender': 1, 'region': 2, 'economic_status': 2,
            'action_id': 7, 'environment_data': 25,
            'work_success': 0.6, 'financial_gain': 80,
            'social_interaction': 0.4, 'stress_level': 0.22
        }
    ]
    
    best_result = llm.find_best_action(test_actions, verbose=False)
    
    print(f"\n最適行動:")
    print(f"  Action ID: {best_result['best_action']['action']['action_id']}")
    print(f"  予測報酬: {best_result['best_action']['predicted_reward']:.4f}")
    print(f"  信頼度: {best_result['best_action']['confidence']:.2%}")
    print(f"  Siri コンテキスト: {best_result['siri_context']}")

    # フィードバック追加
    print("\n【フィードバック統合】")
    print("-" * 70)
    
    feedback_action = {
        'age': 30, 'gender': 0, 'region': 1, 'economic_status': 1,
        'action_id': 6, 'environment_data': 35,
        'work_success': 0.7, 'financial_gain': 80,
        'social_interaction': 0.6, 'stress_level': 0.15
    }
    
    llm.add_feedback(feedback_action, verbose=False)

    # 統計表示
    print("\n【統計情報】")
    print("-" * 70)
    stats = llm.stats()
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")

    print("\n" + sep)


if __name__ == "__main__":
    demo()
