# road_line_ransac

RANSACを用いた白線検出 — 車載カメラ映像からエッジ検出 + RANSAC直線フィッティングで白線を検出します。

## 概要

1. 入力動画を指定フレーム間隔で切り出す
2. Cannyエッジ検出（白色画素フィルタ・ガウシアンブラー付き）
3. エッジ点群に対して RANSAC で直線を繰り返し検出
4. 検出した直線を元フレームに重畳して保存（オプションでリアルタイム表示も可）

全パラメーターはコマンドライン引数で変更可能です。

## ファイル構成

```
road_line_ransac/
├── main.py              # エントリーポイント（引数解析・処理ループ）
├── ransac.py            # RANSACアルゴリズム（直線検出）
├── edge_detection.py    # エッジ検出ユーティリティ
├── video_utils.py       # 動画フレーム取得・描画ユーティリティ
├── requirements.txt     # Python依存パッケージ
└── test_road_line_ransac.py  # ユニットテスト
```

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python main.py <入力動画> [オプション]
```

### 基本例

```bash
# デフォルト設定で実行（10フレームごと、結果を output/ に保存）
python main.py dashcam.mp4

# パラメーターを指定して実行
python main.py dashcam.mp4 \
    -o results/ \
    --frame-interval 5 \
    --canny-low 50 --canny-high 150 \
    --blur-kernel 5 \
    --white-threshold 200 \
    --roi-top 0.5 \
    --ransac-iterations 200 \
    --ransac-threshold 5.0 \
    --ransac-min-inliers 50 \
    --max-lines 2 \
    --display
```

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `input` | ― | 入力動画ファイルパス |
| `-o / --output` | `output` | 出力ディレクトリ |
| `--frame-interval N` | `10` | N フレームごとに1枚処理 |
| `--blur-kernel K` | `5` | ガウシアンブラーのカーネルサイズ（奇数） |
| `--canny-low T` | `50` | Canny 下側閾値 |
| `--canny-high T` | `150` | Canny 上側閾値 |
| `--white-threshold T` | `200` | 白色画素フィルタの輝度閾値（0で無効） |
| `--roi-top R` | `0.5` | 上側マスク割合（0〜1、0.5で下半分のみ処理） |
| `--ransac-iterations N` | `200` | RANSAC の反復回数 |
| `--ransac-threshold D` | `5.0` | インライア判定距離（ピクセル） |
| `--ransac-min-inliers N` | `50` | 直線として採用する最小インライア数 |
| `--max-lines N` | `2` | 1フレームあたり検出する最大直線数 |
| `--display` | off | 処理結果をウィンドウ表示（`q` で終了） |
| `--no-save` | off | ファイル保存を行わない |

## テスト実行

```bash
python -m pytest test_road_line_ransac.py -v
```
