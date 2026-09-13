# DeepSeek V4.1 engram を MXFP4 へ — 2026-09-10

## なぜ

V4.1 をこの host で動かす条件は RAM。expert は既に MXFP4（E2M1 + E8M0 block32）
なので触らない。engram embedding table だけが F8_E4M3 で、2 枚で 203 GB ある。

| | 現状 | MXFP4 化後 |
|---|---:|---:|
| engram table ×2 | 203 GB | **104 GB** |
| expert（据え置き） | 288 GB | 288 GB |
| CPU 側合計 | 491 GB | **392 GB** |
| 実用 RAM | 490 GB | 490 GB |
| 余裕 | **−1 GB** | **+98 GB** |

dense 約 14 GB は GPU。engram を落とさなければ 1 GB 足りずに入らない。

## checkpoint の実測値

`layers.{1,14}.engram.embed`:

| tensor | dtype | shape |
|---|---|---|
| `.weight` | F8_E4M3 | `[384016682, 256]` |
| `.scale` | F8_E8M0 | `[384016682, 8]` |

dim=256、scale が 8 本 / 行なので **block_size=32**。expert と同じ粒度、同じ
scale 形式。だから **scale 経路は一切変えずに payload だけ差し替えられる**。

| | B/row | table |
|---|---:|---:|
| 現状 F8_E4M3 + E8M0 | 264 | 101.4 GB |
| E2M1 + E8M0 block32 | **136** | **52.2 GB**（52%） |
| E2M1 + E8M0 block16 | 144 | 55.3 GB（55%） |

## 上流実装の読み取り

`python/sglang/srt/layers/engram.py`（PR #38798、954 行）:

- `EngramEmbedding` が `weight`（fp8_e4m3 `[rows, dim]`）と
  `scale`（fp8_e8m0 `[rows, dim/FP8_BLOCK_SIZE]`）を持つ
- host table は `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` で有効化。
  layout は `shared`（TP 全 rank で 1 コピー、all-reduce 不要）か
  `private`（rank ごとの shard）。**この host には shared が要る**
- `_HostTable` は `MAP_PRIVATE|MAP_ANONYMOUS` か共有 fd の mmap に
  `MADV_HUGEPAGE`、そのうえで**全ページを事前 fault** する。
  file 裏打ちではないので「必要な行だけ page cache」にはならない。
  `SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE` は 512 MiB huge page を
  取るために checkpoint の page cache を `POSIX_FADV_DONTNEED` で捨てる
- gather は `engram_gather(w_ptr, s_ptr, ids, out, dim, FP8_BLOCK_SIZE)`、
  Triton 80 行

## 実装

**再量子化** `tools/requantize_engram_mxfp4.py`。F8_E4M3 + E8M0 を fp32 へ戻し、
block ごとに amax から `ceil(log2(amax/6))` で E8M0 指数を選び直して E2M1 へ丸め、
nibble を low-first で詰める。合成データの自己検証:

```
bytes/row 264 -> 136 (52%)
max |err|/rowmax 0.2000   mean 0.0036
cosine  min 0.984375   mean 0.993615
exact round trip on E2M1-representable values: OK
```

**この cosine は良くない。** 本プロジェクトが NVFP4 kernel を検証したときの
合格線は 0.999992 で、4 桁悪い。原因は E2M1 の粗さ（仮数 1 bit、
`{0, .5, 1, 1.5, 2, 3, 4, 6}` の 8 段階のみ）。ただし自己検証は E4M3 の全 bit を
一様乱数で作った最悪条件なので、実 table では改善する可能性がある。
**実 checkpoint で測り直してから採否を決める。**

**kernel** `patches/dsv41-engram-gather-mxfp4.py`。上流の gather に
`MXFP4: tl.constexpr` を足しただけで、scale 経路は無改造。payload は
`[rows, dim//2]` を uint8 で読み、low/high nibble に割って `tl.interleave` で
element 順へ戻す。E2M1 の復号は表引きせず算術で行う:

```
code < 4  -> code * 0.5
それ以外  -> (1 + (code & 1) * 0.5) * 2 ** ((code - 4) // 2 + 1)
```

`tools/check_e2m1_decode.py` が 16 code 全部で参照 level 表と一致することを
確認済み（符号付きを含め ALL MATCH）。**GPU 実行検証は未了**（学習が両 GPU を
使用中、21 時以降に空く予定）。

## 残り

1. 本体 475.3 GiB の取得（HDD、進行中）
2. 実 engram table での再量子化と品質実測 ← 採否の分岐点
3. Triton kernel の GPU 実行検証
4. `EngramEmbedding` の `w_bytes` を `n * dim // 2` にする配線
5. kt-kernel への expert 接続（V4.1 は MXFP4 block32、既存の group32 経路）

## 第三案: engram を NVMe 裏打ちにする — 2026-09-10

SNS のタイムラインで見かけた思いつきとして持ち込まれた案。出典としての重みは
無いが、着想は妥当なので測った。engram は n-gram hash で引く条件付き記憶なので
アクセスが疎なはず、ならば file 裏打ちの mmap にして引かれた行だけ page cache に
載せれば、全常駐も再量子化も要らない、という筋。

### 上流実装は逆を選んでいる

`_HostTable._open_shared_fd` は `os.memfd_create`、つまり**匿名 RAM**。`private`
layout も `MAP_ANONYMOUS`。どちらも `MADV_HUGEPAGE` を付けて**全ページを事前
fault** し、さらに `POSIX_FADV_DONTNEED` で checkpoint の page cache を捨てる。
512 MiB huge page を確実に取るための設計で、「必要な行だけ載せる」とは正面から
衝突する。環境変数は 4 つあるがパスを受け取るものは無い。file 裏打ちにするには
実装に手を入れる必要がある。

### 局所性の実測

`tools/engram_locality.py`。hash は
`val = xor_sum(mapped_token * multiplier) % prime + offset` で、(layer, n-gram
size, head) ごとに素数と offset が異なる disjoint な範囲に落ちる。よって行が
衝突するのは**n-gram の内容が再出現したときだけ**で、局所性は n-gram 重複率に
等しい。weights も GPU も要らず tokenizer だけで測れる。

config 実測値: `engram_layer_ids=[1,14]`（2 層）、`engram_max_ngram_size=4`、
`engram_n_heads=8`、table は 768,022,850 行 × 256。
**1 トークンあたり 2 層 × 3 n-gram サイズ × 8 head = 48 行**。

Flash-Next が実際に書いた日本語 1,125 トークンで測定:

| | |
|---|---:|
| 参照 / token | 48 行 |
| 再訪率 | **10.3%** |
| **新規行 / token** | **43.0** |

窓を 256/512/768 と広げても再訪率は 6.5〜7.6% で伸びない。日本語の自然文で
4-gram が再出現することは稀なので、当然の結果。

（最初の測定は `--repeat 40` で同じ文を 40 回繰り返してしまい、再訪率 97.5% と
出た。2 周目以降が全て再訪になっただけの無効な値なので破棄した。）

### page fault の費用

新規行 43 個 = 4 KiB ランダム読み 43 回 / token:

| 置き場所 | ms/token |
|---|---:|
| NVMe（80 us） | **3.44** |
| SATA SSD（150 us） | 6.45 |
| HDD（8 ms） | **343.84** |

**HDD は論外**（3 tok/s 以下）。NVMe なら decode 30〜50 ms に対して 7〜11% の
上乗せで、許容範囲。ただし gather kernel は 48 行を同時に引くので fault は
並列に走る一方、fault の CPU 側処理は直列成分を持つ。実測しないと確定しない。

### 三案の比較

| 案 | RAM | NVMe | 速度 | 品質 |
|---|---:|---:|---|---|
| 上流のまま（FP8 全常駐） | 491 GB | — | 無傷 | 無傷。**1 GB 足りず入らない** |
| **MXFP4 化（全常駐）** | **392 GB** | — | 無傷 | cosine 0.984、要実測 |
| NVMe 裏打ち（FP8 のまま） | 288 GB | 203 GB | +3.4 ms/token | **無傷** |

NVMe 案には別の壁がある。NVMe の空きは 270 GB で、そこには Flash-Next の GGUF
111 GB と各 venv が既に乗っている。engram 203 GB を追加で置く余地は無く、
採るなら GGUF の退避か削除の判断が要る。

**当面は MXFP4 を先に試す。** 速度の代償がゼロで NVMe の再配置も要らず、未知は
品質だけ。本体の取得が終われば実 table で cosine を測れる。悪ければ NVMe 案へ
移る。安い方から潰す。

## 決定: NVMe 裏打ちを採る — 2026-09-10

### MXFP4 は実 table で品質が改善しなかった

`tools/measure_engram_quality.py` が shard を byte offset で部分読みして、
`layers.14.engram.embed` の 20,000 行を実測（95 GiB を torch load せずに済む）。

| | 合成データ | **実 table 20,000 行** |
|---|---:|---:|
| cosine 最小 | 0.984375 | **0.990329** |
| cosine 平均 | 0.993615 | **0.993445** |
| 平均 err/rowmax | 0.0036 | **0.0301** |
| cos < 0.999 の行 | — | **100.00%** |

「実データなら改善するはず」という見込みは外れた。最小値だけ 0.984 -> 0.990 と
上がったが平均は変わらず、平均誤差は 8 倍悪い。全 20,000 行が 0.999 を割る。
engram の値は block 32 の中でダイナミックレンジが広く、E2M1 の 8 段階では
足りない。

しかも **MXFP4 の可否は堂々巡りで確かめられない**。0.7% のズレが生成へどう出るか
はモデルを動かさないと分からず、動かすには RAM が足りず、足りるようにするのが
この量子化そのもの。

### NVMe 裏打ちに切り替えた根拠

| | RAM | NVMe | 速度 | 品質 |
|---|---:|---:|---|---|
| MXFP4 全常駐 | 392 GB | — | 無傷 | cos 0.993、検証不能 |
| **NVMe 裏打ち（FP8 のまま）** | **288 GB** | **203 GB** | +3.4 ms/token（実測由来） | **無傷** |

品質の代償がゼロで、速度の代償だけが残り、その速度は既に測ってある
（43 新規行/token × 80 us）。未知が一つ減る。

### NVMe の確保

`models/dense-scaling-test`（19 GB、dense の 2 ソケット挙動を測るための使い捨て）
と `models/Gemma-4-31B-IT-NVFP4`（31 GB、CPU offload で 1.94 tok/s、実用外）を
ユーザー承認のうえ削除。270 -> **318 GB**。engram 203 GB を置いて 115 GB 残る。
`tools/start-gemma4-31b-nvfp4.sh` は残してあるので、必要なら NVIDIA から再取得。

### 実装

**抽出** `tools/extract_engram_tables.py`。shard から `embed.weight` と
`embed.scale` の生バイトを、`_HostTable` が既に使っている順序（weight のあと
scale）で 1 ファイルへ streaming コピーし、SHA-256 と manifest を残す。
95 GiB の tensor を torch load しないので RSS は chunk サイズのまま。

**file layout** `patches/dsv41-engram-file-layout.patch`（130 行、2 ファイル）。
`_HostTable` に第 3 の layout を足しただけで、既存の `shared` / `private` には
触っていない:

- `_open_file_fd` が `O_RDONLY` で開き、サイズを検証し、
  `POSIX_FADV_RANDOM` を立てる
- mmap は `MAP_SHARED` + `PROT_READ`、`MADV_HUGEPAGE` ではなく **`MADV_RANDOM`**。
  huge page だと 1 行触るたび 512 MiB 引き込むので、file 裏打ちの意味が消える
- 事前 fault はしない
- `pin` は file のとき強制的に無効。`cudaHostRegister` は全ページを fault するので、
  この layout が避けたいコストそのもの
- `_load_rows` は file layout なら即 return。mapping が抽出済み table 自体で
  read-only なので、checkpoint からの copy は不要かつ不可能。size 検証は
  `_open_file_fd` で済んでいる
- `dirty` を立てないので `finish_load` は素通りする（先頭で `if not self.dirty`）
- `_shared` は file も含める。全行を保持するので all-reduce は不要

環境変数 `SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR` を `environ.py` に追加。
`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file` で選ぶ。

MXFP4 の実装（`tools/requantize_engram_mxfp4.py`、
`patches/dsv41-engram-gather-mxfp4.py`）は破棄せず残す。NVMe 裏打ちの
レイテンシが実測で許容外だった場合の退避先。

## 先行実装の調査: 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 — 2026-09-10

同じ壁に同じ順序で当たった人がいた。RTX PRO 6000 96GB×4、TP4/EP4、
`OFFLOAD_MODE=nvme|ram`。**MIT**（Copyright (c) 2026 0xSero）。NOTICE が境界を
明示していて、`adapter/` 配下は MIT、`runtime/flash_mla_sm120.py` だけ SGLang
由来の Apache 2.0、モデル重みは非再配布。参考にしてよい。

README の記述がこちらの見積りを裏づける。RAM モードは「189 GiB の Engram
shard + 24 GiB headroom、256 GiB 以上のホストが実用の出発点」、
「A 128 GB host cannot hold both native Engram tables fully in RAM」。
だから NVMe モードを作った、という経緯。

実測（GPU 4 枚、backbone も KV も全部 VRAM、NVMe モード）:

| 入力×並列 | prefill tok/s | 合計 decode | 1 req あたり |
|---|---:|---:|---:|
| 400k × 1 | 6,129 | **167** | 167 |
| 400k × 6 | 6,178 | 501 | 83 |

**NVMe から engram を引いても単発 167 tok/s 出ている。** こちらの見積り
「43 新規行/token × 80 us = 3.4 ms」が実際に埋もれる程度だという傍証。
ただし構成が違いすぎるので、167 はこちらの上限ではない。

### 実装の要点

`adapter/row_store.cpp`（4,987 B）:

- `O_DIRECT` で page cache を迂回。**RAM 消費を宣言した予算だけに閉じ込める。**
  `mmap` + `MADV_RANDOM` だと page cache の膨張を止められない
- direct-mapped 行キャッシュ。`slot = id % slots`、1 行 264 B 固定、
  256 本の mutex で分割ロック。`slots = budget / 272`
- `read_bytes` は 4 KiB 境界に丸めて `pread`、`O_DIRECT` の整列要件を満たす
- RAM モードは `MAP_POPULATE` + `mlock`、失敗したら起動を止める
- 行が読めなければ `std::abort()`。「Never allow a generation to continue with
  missing/stale rows」

`adapter/engram_backend.py`（5,410 B）— **ここが本質的に賢い**:

- **SGLang を改造しない。** `install(module)` が `EngramEmbedding.__init__` と
  `_owned_rows` を実行時に差し替えるモンキーパッチ。`sitecustomize.py` が仕込む。
  上流が変わっても追随コストが小さい
- **`engram_gather` をそのまま使う。** CPU が行を集めて pinned buffer へ置き、
  GPU へ転送し、**連番 ID** `sequential[:count]` で既存 kernel を呼ぶ。
  kernel を「先頭から順に読むだけ」に退化させることで、FP8 復号と E8M0 scale
  適用を上流実装のまま流用している
- `cudaLaunchHostFunc` で decode graph に組み込む。だから callback 内で
  CUDA API を呼べない（`row_store.cpp` のコメントはそのため）
- `capacity = 1 << (count-1).bit_length()` で staging buffer を 2 冪に丸めて再利用。
  graph capture 前に温めていなければ例外
- `weight_loader` は形だけ検証して**何もロードしない**。table はファイルにある
- `budget // (2 * tp_size)` で層数とランク数に予算を割る

### 方針変更: 自作の file layout パッチは採らない

`patches/dsv41-engram-file-layout.patch`（130 行、`engram.py` 直接改変）は
記録として残すが採用しない。理由は 2 つ。

1. **上流追随コスト。** モンキーパッチなら PR #38798 がマージされても
   `engram.py` の変更に巻き込まれない。直接パッチは当て直しになる
2. **`engram_gather` の流用。** 連番 ID で呼ぶ発想により FP8 復号を自前で
   書かずに済む。こちらのパッチは read-only mmap を GPU から直接 gather させる
   前提で、NVMe 読みには向かない

### そのまま借りない部分

あちらは expert も GPU にあるので、CPU が行集めに専念できる。**こちらは
kt-kernel が expert 計算で CPU を占有する。** 48 行の pread + memcpy が
worker pool とコアを奪い合う可能性がある。ただし NVMe 待ちの間はスレッドが
ブロックするだけなのでコアは空く見込み。**実測で確かめる。**

`std::abort()` で落とす作りも本番には持ち込まない。

TP シャーディングも変える。あちらは `row_store_range` でランクごとに行を分け、
範囲外をゼロにして all-reduce で合成する。行キャッシュがランクごとに別物だから
予算を割る必要があったため。こちらは `O_DIRECT` のファイル読みなので両ランクが
全域を読めばよく、all-reduce が要らない。

出典表記: NVMe direct I/O + 予算付き行キャッシュ + `cudaLaunchHostFunc` 経由の
CUDA graph 組み込みという構成は
`0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000`（MIT）から。
コードを流用する場合は MIT 表記ごと持ち込む。
