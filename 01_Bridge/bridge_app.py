コードを詳しく読んで、修正点とUI改善点をまとめます。

コードを通読しました。修正が必要なバグと、UI上の改善点をまとめます。

---

## 修正点（バグ・ロジックの問題）

**1. 原子インデックス対応の無検証（重大）**
```python
# Phase 3 の平均計算
avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(matched_results))
```
`atoms[i]` が全 conformer で同じ原子を指す保証がない。conformer ごとに原子数やファイル形式が異なる場合、誤った原子間で平均してしまう。`atom['index']` でキー付き辞書に変換して突合すべき。

**2. `matched_results` が ID 順にソートされていない**  
ファイルのアップロード順に依存するため、Boltzmann 重みの計算結果が不定。`sorted(matched_results, key=lambda r: r['id'])` を先頭に入れるべき。

**3. `compose_route` での dispersion 重複追加**  
"B3LYP-D3BJ" を選択すると `active_method` に `empiricaldispersion=gd3bj` が含まれる。さらに `orig_details.get("dispersion")` が非 None なら `compose_route` 内でも追加されるため、Route 文字列に dispersion が二重に入る。

**4. `auto_pad_label` の API が壊れている**  
Analysis Mode での呼び出し:
```python
df_labeled['Atom_Label'].apply(lambda x: auto_pad_label(x, "", ""))
```
`element=""`, `no=""` を渡しているが、`label` が空でない場合のみ実行されるため実害は出ないが、第2・第3引数が無意味で紛らわしい。Backup Mode と共通の関数として設計するか分離すべき。

**5. `sum(weights)` のゼロ除算リスク**  
温度が 0 K に近い場合や全エネルギーが極端に離れている場合に `sum(weights) == 0` となり `ZeroDivisionError`。

**6. `element_symbols` が不完全**  
Ru, Pd, Rh, Pt などの遷移金属が未収録。天然物・医薬品の計算でも Se(34), Br(35) 以外は漏れが多い。辞書を拡張するか、`mendeleev` 等でフォールバックすべき。

**7. セッション状態が入力変更後もリセットされない**  
ファイルを差し替えても `st.session_state.processed_analysis / processed_backup` が `True` のままなので、古いデータのダウンロードボタンが残り続ける。

**8. `frequency_map[fid]` が None を返しうる箇所の未チェック**  
`energy_map[fid]` が None の場合はガードしているが、`imaginary_ids` 抽出時に `r["frequency"]["imaginary"]` へ直接アクセスしており、`parse_frequency_source` が何らかの理由で不完全な辞書を返した場合に KeyError になりうる。

---

## UI 改善点

**1. Phase 番号の命名**  
「Phase 1.5」「Phase 1.6」は後付け感があり、フローが直感的でない。「Frequency Check」「Reoptimization Input」のような機能名ベースの見出しにするか、サイドバーにナビゲーション的なステップ表示を置くと整理される。

**2. Phase 1.6 の設定 UI が情報過多**  
3 カラムのセレクター + Route プレビュー textarea + Link0 textarea + 変位スケール + multiselect が一段に並んでおり圧迫感がある。`st.expander("Advanced settings")` で Route/Link0 を折りたたむだけでかなりすっきりする。

**3. 「Prepare → Download」の 2 ステップが分かりにくい**  
ボタンを押して初めてダウンロードボタンが出現するパターンは UX 上煩雑。`st.download_button` は直接ファイルを生成して渡せるので、Prepare ボタンを廃止して Download ボタン 1 つに統一できる。

**4. データエディタが全原子を表示する**  
大分子では何百行にもなる。ラベル入力用のフィルタ（例: element でフィルタ）か、「ラベルを入力した原子のみ表示」トグルがあると実用的。

**5. 絵文字が多い**  
🌉🧭🔍🫨🚀 などの絵文字が各セクション見出しに多用されており、プロフェッショナルな用途（論文・学術）には重い印象。絵文字を取り除くか 1〜2 個に絞ると全体が締まる。

**6. ダウンロードボタンが各フェーズに点在**  
CSV が Frequency Check、Raw Tensors、Analysis、Backup と 4 箇所に分散している。最終フェーズに「Export Summary」としてまとめるか、サイドバーにまとめた方が見通しが良い。

**7. `st.info()` の待機メッセージが下部に流れる**  
NMR ファイルがない場合のメッセージが Phase 1.5/1.6 の長いセクションの後に来るため見つけにくい。ページ上部のファイルアップローダー直下に `st.status()` や `st.caption()` でステータスを示すと分かりやすい。

**8. サイドバーが温度設定のみで未活用**  
グローバル設定（温度、デフォルト Link0、element_symbols の上書きなど）をサイドバーに集約すると、メインエリアがフローに集中できる。
