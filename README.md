# 画像が消えたら止まり、戻ったら片付けを再開する

Physical AI応用編1の最終課題。

公開済みのSmolVLAで「テーブル中央の黒い器を皿へ移す」ロボットに、
カメラ入力の異常監視・停止・復旧処理を追加した。
AIが画像を見られないまま動作を続ける場合と、画像異常で移動指令を止め、
復旧後に現在の状態から再計画する場合を、同じ条件で比較する。

## 既存部分と今回の追加

**既存部分**は公開モデル、LIBEROの基本タスク、LIBERO-plus/robosuite/MuJoCo環境、
コンペ用に使っていた推論アダプター。モデルの追加学習はしていない。
固定の英語指示を使用し、任意の日本語指示を受け取るUIは実装していない。

**今回の追加**は次の処理と実証である。

1. カメラ画像の欠落・形状不正・非有限値・全黒画像を毎制御ステップで監視。
2. 異常中は腕6成分の移動指令を0にし、グリッパーは直前の指令を維持。
3. 正常画像が3フレーム続いたら、異常前の動作列を破棄。
4. 現在の画像・腕の状態からAIが動作を再予測してタスクを再開。
5. 同一条件での対比較、実際の入力画像と動作の同期撮影、結果・資料の生成。

異常検知・停止判断はルール処理であり、AIを追加学習して身につけさせた機能ではない。
SmolVLAは、画像・英語指示・腕の状態から実行する動作を予測する役割を担う。

## 実験条件と観測結果

1タスク・保存初期状態index 0・seed 42、各条件1回。
制御20 Hz、開始前10ステップ静定、最大280ステップ。
異常条件では**手元カメラ画像だけ**をstep 20〜59（1秒以上3秒未満）に全黒へ置換。
前方カメラ、物理シーン、AIモデル重み、初期配置・seedをそろえた。

| 条件 | タスク達成 | シミュレーション時間 |
| --- | --- | ---: |
| 画像正常（参考） | 達成 | 4.70秒 |
| 画像欠損・対策なし | 未達成 | 14.00秒で打切り |
| 画像欠損・停止復旧あり | 達成 | 6.70秒 |

停止復旧ありではstep 20（1.00秒）に停止し、step 60（3.00秒）に画像復旧。
3フレームの正常受信を確認してstep 62（3.10秒）に古い計画を破棄・再推論し、再開した。
停止中42ステップの保存ログで、腕6成分の指令がすべて0、グリッパー指令が停止前と
同じであることを確認した。

これは1例の実証であり、一般的な成功率改善や統計的な優位性は未確認。
フリーズ画像など全故障を検知するものではなく、実機の安全性・物理的な完全停止・
衝突回避は検証していない。成功はLIBEROのタスク達成条件で判定し、
コンペ独自の衝突条件は適用していない。

## 提出物

- `output/PAI最終課題_YOUR_ACCOUNT.mp4`：説明・正常時の動作・小窓付き比較・結果と限界を含む動画。
- `output/説明資料_PAI最終課題_YOUR_ACCOUNT.pdf`：概要・目的・AIの役割・工夫・実測・考察・出典。
- `output/pai-final-2026-source.zip`：この主題のコードと実測ログ。
- `output/video_description.txt`：YouTubeタイトルと概要欄の文案。
- `SUBMISSION.md`：公開・アップロード・提出の手順。

GitHub・YouTubeへの公開とOmniCampusへの提出は、ローカル成果物の生成とは別の作業である。

## ソース構成

| ファイル | 役割 |
| --- | --- |
| `camera_recovery.py` | 今回追加した異常監視・停止・復旧処理 |
| `experiments/probe_camera_recovery.py` | 欠損を注入し、処理の有無で比較。任意で正常時も収録 |
| `experiments/build_camera_preview.py` | 実際の手元画像の小窓を付けた同期比較 |
| `build_camera_submission.py` | 提出動画・PDF・コードZIP・説明文を生成 |
| `run_demo.py` | 再利用した実行・撮影の共通処理。比較実行からplain設定で呼び出す |
| `offline_policy.py` | 再利用したSmolVLA入力整形・推論・動作列の実行処理 |
| `prepare_assets.py` | 公開モデル・環境・アセットを取得し入力定義を対応させる |
| `results/camera-recovery/{normal,unprotected,guarded}/` | 条件別の実測結果・動作・入力画像ログ |
| `pyproject.toml`, `uv.lock` | Python依存関係 |

モデル重み、第三者ソース、データセット、認証情報はソースZIPに同梱しない。
ローカルに残る旧平滑化比較のファイルは、今回の提出用ソースZIPには含めない。

## 収録済み結果から提出物を再生成

Python 3.12、uv、FFmpeg、日本語フォントを用意する。
`YOUR_ACCOUNT`は提出に使うOmniCampusアカウント名またはIDへ置き換える。

```bash
uv sync
uv run python build_camera_submission.py --account YOUR_ACCOUNT
```

この操作は保存済みの映像・入力画素・結果から成果物を生成し、モデルの再学習・再実行はしない。
macOSの既定フォントは動画がヒラギノ角ゴシック、PDFがArial Unicode。
他の環境では`--font /path/to/JapaneseFont.ttc --pdf-font /path/to/NotoSansJP.ttf`を指定する。
動画はH.264 / 1280×720 / 20 fps、無音・日本語字幕。
PDFは日本語TrueTypeフォントを埋め込む。

## シミュレーションから再実行

macOSではApple SiliconのMPSを使用。本実験の確認環境はmacOS 14.5 / MPS。
新規環境の一括セットアップやLinuxでの再実行は今回未検証。
OS依存はFFmpegとImageMagick。初回アセット取得に10 GB以上の空き容量とネット接続が必要。

```bash
# macOSでOS依存がない場合
brew install ffmpeg imagemagick
uv sync
uv run python prepare_assets.py
uv run python experiments/probe_camera_recovery.py \
  --capture-inputs --include-normal --output-root outputs/camera-submission
uv run python build_camera_submission.py \
  --input outputs/camera-submission --account YOUR_ACCOUNT
```

シミュレーション中は学習・W&B送信・モデルのオンライン取得を行わない。
LIBEROの設定は各出力先配下に作り、ホームディレクトリの設定には触れない。

## 映像と実際の入力画像

`wrist_inputs.npz`に、全黒への置換後・行動実行前のRGB画像を毎ステップ記録する。
向きはモデル前処理と同じ180度回転とし、小窓のステップiと全体映像のフレームiを対応させる。
モデルが画像を参照するのは`prediction_steps`に記録した動作列の再予測時だけであり、
毎フレーム推論しているわけではない。停止中は監視処理が復旧を確認し、AIは呼び出さない。

映像は20 fpsで制御20 Hzに対応するシミュレーション時間で等速再生する。
推論待ち時間は含まず、倍速編集はしない。完了後に全体映像と最終入力を静止表示したことを明記する。

## 公開依存と固定revision

- [SmolVLAモデル](https://huggingface.co/lerobot/smolvla_libero_plus)：`7bb70aa5bc92b82c9239142775d3a173103567ff`
- [VLMアセット](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)：`7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)：`4976dc30028e805ff8094b55501d532c48fec182`
- [LIBERO-plusアセット](https://huggingface.co/datasets/Sylvest/LIBERO-plus)：`dd2bd61b7d9a6fef1abc52d606e983b41886a149`
- [LeRobot](https://github.com/huggingface/lerobot) 0.6.0
- [MuJoCo](https://github.com/google-deepmind/mujoco) 3.7.0

公開モデルの設定ファイルには古いカメラ名・状態次元が残るため、
`prepare_assets.py`でfront/wristの2視点・状態8次元へ対応させる。モデルのテンソル重みは変えない。
第三者の重み・ソース・アセットは提供元から取得し、それぞれの利用条件に従う。
