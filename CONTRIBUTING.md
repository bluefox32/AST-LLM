# 貢献ガイド

このドキュメントは、AST-LLM Unified プロジェクトへの貢献方法を説明します。

---

## 🤝 どのように貢献できますか？

### 1. バグ報告

バグを見つけた場合は、GitHub Issues で報告してください。

**テンプレート**:
```
[BUG] 簡潔なタイトル

## 環境
- Python version: 3.9
- OS: macOS 12.0
- NumPy version: 1.22.0

## 現象
何が起きたのか説明してください

## 再現手順
1. ...
2. ...
3. ...

## 期待する動作
どうなるべきかを説明

## 実際の動作
実際に何が起きたか（エラーメッセージなど）

## スクリーンショット
関連があれば添付
```

### 2. 機能提案

新機能の提案は Issues で `[FEATURE]` タグを付けて投稿してください。

**テンプレート**:
```
[FEATURE] 機能名

## 概要
新機能の概要説明

## 使用例
```python
llm = ASTLLMUnified()
# 使用例のコード
```

## 利点
この機能がもたらす利点

## 代替案
他の実装案があれば記載
```

### 3. ドキュメント改善

README や docstring の改善も歓迎します。

- 誤字修正
- 例の追加
- 説明の明確化
- 翻訳

---

## 🔧 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/ast-llm-unified.git
cd ast-llm-unified

# 仮想環境を作成
python -m venv venv

# 仮想環境をアクティベート
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt

# 開発用パッケージもインストール（オプション）
pip install pytest pytest-cov black flake8 mypy
```

---

## 💻 コード貢献のワークフロー

### ステップ 1: フォークとクローン

```bash
# GitHub で "Fork" をクリック
git clone https://github.com/yourname/ast-llm-unified.git
cd ast-llm-unified

# upstream を追加
git remote add upstream https://github.com/original/ast-llm-unified.git
```

### ステップ 2: ブランチを作成

```bash
# 最新の upstream/main から
git fetch upstream
git checkout -b feature/my-new-feature upstream/main

# または
git checkout -b fix/bug-fix upstream/main
```

### ステップ 3: コード修正・追加

```bash
# ファイルを編集
# テストを追加

# 変更を確認
git diff

# ステージング
git add .

# コミット
git commit -m "feat: 短い説明"
# または
git commit -m "fix: バグ修正の説明"
```

### ステップ 4: テスト実行

```bash
# ユニットテスト
python test_ast_llm.py

# コード品質チェック
flake8 ast_llm_unified.py
black --check ast_llm_unified.py
mypy ast_llm_unified.py
```

### ステップ 5: Push と PR

```bash
# upstream で更新確認
git fetch upstream

# ローカルブランチを更新（競合がないか確認）
git rebase upstream/main

# 自分のリモートに Push
git push origin feature/my-new-feature

# GitHub で Pull Request を作成
```

---

## ✅ コード品質基準

### 必須ルール

1. **900行以下を維持**
   - コンパクトさはこのプロジェクトの売り
   - 大きな追加は小分けに

2. **冪等性・不変条件**
   - 新しい `TransformPass` には冪等性チェック必須
   - validator 関数を実装

3. **Pure Python + NumPy のみ**
   - 外部ライブラリ追加は禁止
   - OS互換性を保つ

4. **ハッシュチェーンの完全性**
   - AST ノード操作時は改ざん検知を考慮

### コード例（良い例）

```python
def my_new_pass(node: ASTNode, scope: Scope) -> str:
    """
    新しい変換パス
    
    Args:
        node: 変換対象のASTノード
        scope: スコープコンテキスト
    
    Returns:
        str: 変換結果
    
    Note:
        決定的な実装（冪等性が必須）
    """
    # 実装
    return result

def validate_my_pass(node: ASTNode, output: str) -> list[str]:
    """出力の不変条件をチェック"""
    errors = []
    if not output:
        errors.append("出力が空")
    if not isinstance(output, str):
        errors.append("出力が文字列でない")
    return errors

# パイプラインに追加
pipeline.add_pass(TransformPass(
    name="my_new_pass",
    fn=my_new_pass,
    validator=validate_my_pass
))
```

### コード例（悪い例）

```python
# ❌ ランダム性（冪等性違反）
import random
def bad_pass(node, scope):
    return str(random.random())  # 毎回異なる

# ❌ 外部ライブラリ依存
import torch  # NumPy以外は禁止
def bad_pass(node, scope):
    return torch.tensor([1, 2, 3])

# ❌ グローバル状態変更
global_state = {}
def bad_pass(node, scope):
    global_state['modified'] = True  # 副作用あり
    return str(node)
```

---

## 📝 ドキュメント更新

### README を更新する場合

```markdown
### 新しい API の追加

#### メソッド: `new_method()`

説明...

##### パラメータ

- `param1` (str): 説明
- `param2` (int, optional): 説明。デフォルト: 100

##### 返り値

`dict` - キーと値の説明

##### 例

\`\`\`python
result = obj.new_method(param1="value", param2=50)
print(result)
\`\`\`

##### 注記

追加情報があれば記載
```

---

## 🧪 テスト作成ガイド

新機能を追加したら、`test_ast_llm.py` に テストを追加してください。

### テストテンプレート

```python
def test_my_new_feature():
    """
    新機能のテスト
    
    テストする内容:
      - 基本的な動作
      - エッジケース
      - エラーハンドリング
    """
    # セットアップ
    obj = MyClass()
    
    # 実行
    result = obj.my_method()
    
    # 検証
    assert result == expected_value, "説明メッセージ"
    assert isinstance(result, expected_type)
    
    print("✓ テストパス")
```

### テスト実行

```bash
# 全テスト実行
python test_ast_llm.py

# 特定のテストだけ実行（pytest使用時）
pytest test_ast_llm.py::test_my_new_feature -v
```

---

## 📋 PR チェックリスト

PR 提出前に確認してください：

- [ ] ブランチ名が適切（`feature/...` または `fix/...`）
- [ ] コミットメッセージが明確
- [ ] 全テストが通過
- [ ] コード品質チェックを実施
- [ ] ドキュメント更新済み
- [ ] 900行以下を維持
- [ ] 外部ライブラリ追加なし
- [ ] 冪等性・不変条件がある場合は実装済み

---

## 🔄 レビュープロセス

1. **自動チェック**
   - GitHub Actions で CI/CD 実行
   - テスト・コード品質チェック

2. **マニュアルレビュー**
   - メンテナーがコードレビュー
   - 質問・修正リクエスト

3. **マージ**
   - 承認後、自動マージまたは手動マージ

---

## 📌 コミットメッセージ規約

Conventional Commits を採用しています：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント変更
- `style`: コード形式（セマンティック変化なし）
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト追加・修正
- `chore`: ビルド・依存関係など

### 例

```
feat(attention): add multi-head attention support

- 複数ヘッドの並列計算に対応
- パフォーマンスが20%向上

Closes #123
```

```
fix(tokenizer): fix encoding issue with unicode

修正内容の詳細...

Fixes #456
```

---

## 🆘 質問・サポート

- **一般的な質問**: Discussions
- **バグ・機能**: Issues
- **コード相談**: Pull Request のコメント

---

## 📚 参考資料

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 Python Style Guide](https://pep8.org/)

---

## 🎉 貢献ありがとうございます！

このプロジェクトの成長は、皆さんの貢献のおかげです。

**Happy Contributing! 🚀**

