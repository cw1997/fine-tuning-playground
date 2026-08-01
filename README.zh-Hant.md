# Fine-Tuning Playground — 微調遊樂場

一套面向教學的程式碼，專為 **Qwen3 / Qwen3.5** 僅解碼器（decoder-only）模型（例如 **Qwen3-4B**、**Qwen3.5-4B**）的監督式微調（Supervised Fine-Tuning, SFT）而設計，採用 **QLoRA / LoRA** 技術。專案基於 Hugging Face 的生態系統建構，完整訓練流程集中在單一腳本 `src/train.py` 中，由上而下順序執行，所有設定皆以命令列參數傳入。

---

## 目錄

- [專案概述](#專案概述)
- [功能特色](#功能特色)
- [硬體需求](#硬體需求)
- [安裝方式](#安裝方式)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [設定參數](#設定參數)
- [建議訓練預設組合](#建議訓練預設組合)
  - [平台與作業系統](#平台與作業系統)
  - [VRAM 與模型規模（QLoRA）](#vram-與模型規模qlora)
  - [資料集規模指引](#資料集規模指引)
  - [具名預設組合](#具名預設組合)
  - [預設腳本](#預設腳本)
  - [各預設的手動命令](#各預設的手動命令)
- [資料格式](#資料格式)
  - [ChatML 格式](#chatml-格式)
  - [Alpaca 格式](#alpaca-格式)
  - [純文字格式](#純文字格式)
- [使用方式](#使用方式)
  - [基本訓練](#基本訓練)
  - [使用 Alpaca 資料集訓練](#使用-alpaca-資料集訓練)
  - [從 Hugging Face Hub 載入資料集](#從-hugging-face-hub-載入資料集)
  - [推論 — 互動式（預設）](#推論-互動式預設)
  - [推論 — 單次執行（--no_interactive）](#推論-單次執行no_interactive)
  - [啟用思考模式推論](#啟用思考模式推論)
  - [檢查模型](#檢查模型)
  - [檢查分詞器](#檢查分詞器)
- [進階用法](#進階用法)
  - [自訂 LoRA 目標模組](#自訂-lora-目標模組)
  - [推送至 Hugging Face Hub](#推送至-hugging-face-hub)
  - [完整精度 LoRA vs QLoRA](#完整精度-lora-vs-qlora)
  - [訓練時啟用思考模式](#訓練時啟用思考模式)
- [輸出產物](#輸出產物)
- [建議與最佳實踐](#建議與最佳實踐)
- [疑難排解](#疑難排解)
- [授權條款](#授權條款)

---

## 專案概述

**Qwen3** 與 **Qwen3.5** 是阿里巴巴 Qwen 團隊推出的稠密語言模型。4B 級別的型號支援長脈絡、分組查詢注意力，並可選擇啟用連鎖思考（chain-of-thought）**思考模式**（透過對話模板中的 `<think>` 標籤）。

本專案提供一套完整的微調流程，讓您能將這些模型調整為符合自身任務與領域需求的版本。其核心技術包括：

- **QLoRA**（4-bit NormalFloat 量化）：將記憶體需求降至約 6 GB VRAM，使消費級 GPU（如 RTX 3060 / 4060 / 4070）也能順利進行微調。
- **`trl.SFTTrainer`**：提供監督式微調的完整實作，內建對話模板整合與評估迴圈。
- **全部 7 個線性投影層**（`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）作為 LoRA 目標，確保在 4B 規模模型上達到最佳適應效果。

---

## 功能特色

- **QLoRA / LoRA 雙模式支援** — 使用 4-bit 量化可在 6 GB GPU 上訓練，關閉量化則可使用完整 FP16 LoRA。
- **多種資料格式** — 支援 ChatML（原生格式）、Alpaca（instruction/input/output）以及純文字格式。
- **Hugging Face Hub 整合** — 直接從 Hub 載入資料集，並將訓練完成的適配器推送回 Hub。
- **思考模式** — 可選擇在訓練與推論時啟用 Qwen3 的 `<think>` 推理標籤。
- **命令列優先設計** — 所有超參數皆可透過命令列旗標設定，無需撰寫 YAML 設定檔。
- **CPU 備援** — 若無 CUDA，訓練與推論會自動改以 CPU 全精度執行（速度極慢，僅建議用於冒煙測試）。
- **單檔訓練腳本** — `src/train.py` 由上而下執行完整 SFT 流程，適合講解微調各步驟。
- **模型檢查** — `src/inspect_model.py` 輸出模型內部的標準化報告（參數量、詞表大小、嵌入維度、層級結構），無需準備資料集。
- **分詞器檢查** — `src/inspect_tokenizer.py` 顯示句子被切分為哪些 token、每個 token 的詞表編號，以及每個 token 的嵌入向量（或單一 token 的 index 位置與向量）。
- **微調前後對比** — 可先測試微調前的基底模型，再與微調後的模型進行逐題對比，量化改善幅度。
- **互動式推論** — `src/inference.py` 預設在載入模型後進入提示詞輸入迴圈；使用 `--no_interactive` 可改為單次批次推論後退出。
- **評估資料分割** — 自動將資料集分割為訓練集與測試集，並根據評估損失選取最佳模型。
- **梯度檢查點** — 預設啟用，可降低訓練過程中的記憶體使用量。

---

## 硬體需求

| 設定 | 最低 VRAM | 建議 GPU |
|---|---|---|
| QLoRA 4B（4-bit，batch 1–2） | ~6–8 GB | RTX 3060 / 4060 / 4070、T4 |
| QLoRA 9B（4-bit，batch 1） | ~10–12 GB | RTX 4070 Ti / 5070 Ti（筆電）、3080 12 GB |
| QLoRA 9B（4-bit，batch 2） | ~20–24 GB | RTX 4090 / 3090 24 GB、A5000 |
| LoRA 4B（16-bit，batch 1） | ~12 GB | RTX 3080 / 4070 Ti / 4080、A10 |
| CPU（無量化，float32） | — | 任何 CPU（極慢，僅冒煙測試） |

強烈建議使用 GPU 訓練。在無 CUDA 的機器上，流程會自動關閉 4-bit 量化並以 CPU `float32` 執行。可用 `--device gpu` 或 `--device cpu` 覆寫自動偵測。各 GPU 等級的現成命令請見[建議訓練預設組合](#建議訓練預設組合)。

---

## 安裝方式

建議使用 **Python 3.11 或 3.12** 以獲得最佳 CUDA wheel 相容性。較新的 Python 版本在 PyPI 上可能只有 CPU 版 PyTorch。

### 方式一 — Miniconda（建議）

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

conda create -n finetune python=3.11 -y
conda activate finetune

pip install -r requirements.txt
```

### 方式二 — venv

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 輔助腳本

```bash
bash scripts/install_deps.sh                 # Linux / macOS / Git Bash
powershell -File scripts/install_deps.ps1    # Windows PowerShell

# 列出訓練預設（VRAM / 模型 / 資料集規模）
bash scripts/train_preset.sh --list          # Linux / macOS / Git Bash
powershell -File scripts/train_preset.ps1 -List
```

驗證 PyTorch 是否能看到 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`requirements.txt` 包含 CUDA 版 PyTorch（cu128）與全部專案依賴。若僅需 CPU 版（4B 模型會非常慢），請先從 PyPI 安裝 PyTorch，再安裝其餘套件（不含 `bitsandbytes`）。

核心套件：

| 套件 | 最低版本 | 用途 |
|---|---|---|
| `torch` | >= 2.4.0, < 2.12 | 深度學習框架（`requirements.txt` 預設 CUDA cu128） |
| `transformers` | >= 4.47.0 | 模型載入與分詞（Qwen3 支援） |
| `peft` | >= 0.13.0 | LoRA / QLoRA 適配器設定 |
| `trl` | >= 0.9.0 | 監督式微調的 SFTTrainer |
| `accelerate` | >= 1.0.0 | 分散式訓練支援 |
| `bitsandbytes` | >= 0.43.0 | 4-bit 量化（QLoRA） |
| `datasets` | >= 3.0.0 | 資料集載入與處理 |

---

## 快速開始

依下列步驟設定環境並開始訓練：

```bash
# 1. 複製並進入倉庫
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

# 2. 建立並啟用 conda 環境（或使用 venv — 見[安裝方式](#安裝方式)）
conda create -n finetune python=3.11 -y
conda activate finetune

# 3. 安裝依賴套件
pip install -r requirements.txt

# 4. 確認 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 5. 使用內附的臺師大資料集（約 1.2K 筆 ChatML 範例）進行訓練
bash scripts/train_preset.sh gpu-12gb-4b
# 或手動執行：
# python src/train.py --model_id Qwen/Qwen3.5-4B --dataset_path ./data/ntnu_dataset.jsonl --output_dir ./models/ntnu/gpu-12gb-4b

# 6. 以基底模型推論（微調前）— 儲存輸出以便對比
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "台灣師范大學的地址是什麼" --no_interactive

# 7. 以微調後模型推論 — 與步驟 6 對比
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./models/ntnu/gpu-12gb-4b --prompt "台灣師范大學的地址是什麼" --no_interactive

# 8. 以微調後模型進行互動式對話（輸入 quit 退出）
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./models/ntnu/gpu-12gb-4b
```

詳細訓練選項與資料格式請見[使用方式](#使用方式)。

---

## 專案結構

```
fine-tuning-playground/
├── requirements.txt          # Python 套件依賴列表
├── src/                      # 原始碼腳本
│   ├── train.py              # 完整 SFT 流程（單檔、由上而下）
│   ├── inference.py          # 互動式推論 + 微調前/後/對比模式
│   ├── inspect_model.py      # 模型內部檢查（參數量、詞表、維度、設定）
│   └── inspect_tokenizer.py  # 分詞器檢查（token 片段、編號、嵌入向量）
├── data/                     # 訓練資料集與產生器
│   ├── ntnu_dataset.jsonl    # 內附臺師大 ChatML 資料集（約 1.2K 筆）
│   ├── full_dataset.jsonl    # 臺師大 + 全臺分類彙整資料集（約 1.7K 筆）
│   ├── generate_ntnu_dataset.py  # 彙整 ntnu/ + taiwan/ 為 full_dataset.jsonl
│   ├── restructure_dataset.py    # 將 ntnu_dataset.jsonl 拆成分類檔
│   ├── ntnu/                 # 臺師大分類 JSONL 檔
│   ├── taiwan/               # 臺灣分類 JSONL 檔
│   └── _archive/             # 舊版紀錄產生模組
├── models/                   # 微調後的 LoRA 適配器（已加入 .gitignore）
├── scripts/                  # install_deps.* / train_preset.*
├── README.md                 # 英文文件
└── README.zh-Hant.md         # 本文件（繁體中文）
```

### 各檔案說明

**`src/train.py`** — 完整的監督式微調流程，全部寫在一個檔案中。依序執行：SSL 修復 → 解析命令列參數 → 偵測裝置 → 載入分詞器 → 載入基座模型（可選 4-bit）→ 套用 LoRA → 載入並格式化資料集 → 設定 `SFTTrainer` → 訓練 → 儲存適配器。所有超參數皆為 CLI 旗標，無獨立設定模組。

**`src/inference.py`** — 獨立推論腳本，支援基底模型、微調後模型或並排對比。預設在載入模型後進入互動式提示詞迴圈（`You:` 提示；輸入 `quit` / `exit` / `q` 退出）。CLI 旗標：`--mode`（`base` / `finetuned` / `compare`）、`--model_id`、`--adapter_path`、`--prompt`、`--no_interactive`、`--load_in_4bit`、`--use_thinking`、`--max_new_tokens`、`--temperature`、`--top_p`、`--device`、`--torch_dtype`。

**`src/inspect_model.py`** — 模型檢查腳本，輸出模型內部的詳細標準化報告：總/可訓練/凍結參數量、token 嵌入形狀與記憶體估算、詞表大小、層級結構、注意力與 FFN 設定、RoPE 參數，以及分詞器中繼資料（特殊 token、padding 方向、chat template）。適合在訓練前驗證下載的模型。CLI 旗標：`--model_id`、`--load_in_4bit`、`--torch_dtype`、`--device`、`--no_load`（僅輸出設定與分詞器）。

**`src/inspect_tokenizer.py`** — 分詞器檢查腳本，輸出句子被切分為哪些 token（附每個 token 的詞表編號），並自模型的輸入嵌入層取出每個 token 的嵌入向量（形狀、dtype、範數與數值）。也支援以 `--token`（字串）或 `--token_id`（編號）查詢單一 token 的 index 位置與向量。byte-level token 會以可讀形式顯示（GPT 風格的 `Ġ` 空格前綴會轉為空格）。CLI 旗標：`--model_id`、`--sentence`、`--token`、`--token_id`、`--vector_limit`（每個向量輸出幾個元素；`0` = 全部）、`--load_in_4bit`、`--torch_dtype`、`--device`、`--no_model`（僅輸出 token 化與編號）。

**`data/`** — 包含內附的 `ntnu_dataset.jsonl`（約 1,252 筆關於國立臺灣師範大學的 ChatML 紀錄）、彙整後的 `full_dataset.jsonl`（約 1.7K 筆，涵蓋臺師大與全臺各大專院校、縣市），以及管理這些資料的腳本：`restructure_dataset.py` 將 `ntnu_dataset.jsonl` 拆分為 `ntnu/` 與 `taiwan/` 下的分類 JSONL 檔（並為全臺各大學與縣市產生新資料），`generate_ntnu_dataset.py` 再將這些分類子目錄彙整為單一的 `full_dataset.jsonl` 訓練檔。`_archive/` 存放舊版紀錄產生模組（`ntnu_extended_records.py`、`ntnu_massive_records{1-4}.py`）供參考。

**`scripts/train_preset.sh` / `scripts/train_preset.ps1`** — 針對常見 GPU 等級與模型規模的具名訓練預設。可用 `DATASET_PATH=./data/my.jsonl` 覆寫資料集路徑。

---

## 設定參數

### 訓練（`src/train.py`）

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face 模型識別碼（GPU 建議使用 `Qwen/Qwen3.5-4B` 或 `Qwen/Qwen3-4B`） |
| | `load_in_4bit` | `True` | 啟用 4-bit NF4 量化（CPU 上會自動關閉） |
| | `torch_dtype` | `bfloat16` | 計算精度（CPU 上會改為 `float32`） |
| **LoRA** | `lora_r` | `16` | LoRA 秩（rank） |
| | `lora_alpha` | `32` | LoRA 縮放因子（通常設為 2 × r） |
| | `lora_dropout` | `0.05` | LoRA 層的 dropout 比率 |
| | `target_modules` | 全部 7 個線性層 | 逗號分隔的模組名稱（例如 `q_proj,k_proj,v_proj,o_proj`） |
| **資料** | `dataset_path` | （必填） | 本機 JSON/JSONL 檔案、JSONL 檔目錄（遞迴掃描），或 HF 資料集名稱（本機路徑須以 `./`、`.` 或 `/` 開頭） |
| | `dataset_format` | `chat` | 輸入格式：`chat`、`alpaca` 或 `text` |
| | `test_split` | `0.05` | 保留為評估集的比例（0 = 不評估） |
| | `max_seq_length` | `4096` | 每個範例的最大 token 數 |
| | `use_thinking` | `False` | 啟用思考模式 |
| **訓練** | `learning_rate`（`--lr`） | `2e-4` | 峰值學習率 |
| | `num_epochs`（`--epochs`） | `3` | 訓練週期數 |
| | `per_device_batch_size`（`--batch_size`） | `2` | 每張 GPU 的批次大小 |
| | `gradient_accum_steps` | `8` | 梯度累積步數 |
| | `warmup_ratio` | `0.03` | 線性 warmup 比例 |
| | `logging_steps` | `10` | 每隔 N 步記錄一次指標 |
| | `save_steps` | `200` | 每隔 N 步儲存一次檢查點 |
| | `output_dir` | `./models/qwen3-4b-finetuned` | 輸出目錄 |
| **硬體** | `device` | `None`（自動） | 運算裝置：`gpu`、`cpu` 或自動偵測 |
| **Hub** | `push_to_hub` | `False` | 將適配器推送至 HF Hub |
| | `hub_model_id` | `""` | Hub 上的目標儲存庫名稱 |

### 推論（`src/inference.py`）

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3.5-0.8B` | 基底模型（須與微調時使用的模型一致） |
| | `mode` | `base` | `base`、`finetuned` 或 `compare` |
| | `adapter_path` | — | LoRA 適配器目錄（`finetuned` / `compare` 模式必填） |
| | `load_in_4bit` | `True` | 推論時使用 4-bit 量化 |
| | `torch_dtype` | `bfloat16` | 計算精度 |
| | `device` | `None`（自動） | `gpu`、`cpu` 或自動偵測 |
| **生成** | `prompt` | — | 單次測試提示（省略則跳過批次測試） |
| | `max_new_tokens` | `2048` | 最大生成 token 數 |
| | `temperature` | `0.7` | 取樣溫度 |
| | `top_p` | `0.9` | 核取樣（nucleus sampling）閾值 |
| | `use_thinking` | `False` | 啟用思考模式 |
| | `no_interactive` | `False` | 批次推論後直接退出，不進入互動迴圈 |

### 檢查（`src/inspect_model.py`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-0.8B` | 要檢查的 Hugging Face 模型識別碼 |
| `load_in_4bit` | `False` | 以 4-bit 量化載入權重（查看量化後的參數量） |
| `torch_dtype` | `float32` | 載入權重時使用的計算精度 |
| `device` | `None`（自動） | `gpu`、`cpu` 或自動偵測 |
| `no_load` | `False` | 僅輸出設定與分詞器報告，不載入權重 |

### 分詞器檢查（`src/inspect_tokenizer.py`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-0.8B` | 要檢查的 Hugging Face 模型識別碼 |
| `sentence` | — | 要分詞的句子；輸出每個 token 片段、編號與向量 |
| `token` | — | 要查詢的單一 token 字串（index 位置與向量） |
| `token_id` | — | 要查詢的單一 token 編號（token 字串與向量） |
| `vector_limit` | `8` | 每個 token 輸出幾個向量元素（`0` 輸出完整向量） |
| `load_in_4bit` | `False` | 以 4-bit 量化載入模型 |
| `torch_dtype` | `float32` | 載入模型時使用的計算精度 |
| `device` | `None`（自動） | `gpu`、`cpu` 或自動偵測 |
| `no_model` | `False` | 僅輸出 token 化與編號，不載入模型（無向量） |

`--sentence`、`--token`、`--token_id` 三者至少需提供其一。

> **備註：** `src/train.py` 與 `src/inference.py` 的 `--model_id` 預設皆為 `Qwen/Qwen3.5-0.8B`（適合 CPU 冒煙測試）。GPU 訓練請傳入 `--model_id Qwen/Qwen3.5-4B` 或 `Qwen/Qwen3-4B`，推論時也須使用**相同**的 `--model_id`。在 `src/inference.py` 中，省略 `--prompt` 會跳過批次測試並直接進入互動迴圈；搭配 `--no_interactive` 且未提供 `--prompt` 時，會使用兩條內建的 NTNU 測試提示。

---

## 建議訓練預設組合

以下預設均假設使用 **QLoRA（4-bit）**、**`bfloat16`**、**ChatML 格式**（`--dataset_format chat`），且**有效批次大小為 16**（`per_device_batch_size × gradient_accum_steps`）。所有預設皆啟用 5% 評估分割，並依 eval loss 儲存最佳檢查點。

內附的臺師大資料集（`./data/ntnu_dataset.jsonl`）約含 **1,252** 筆範例 — 預設使用 **3 epoch**、**lr = 2e-4**。若使用較大的自訂資料集，請搭配下方[資料集規模指引](#資料集規模指引)調整。

### 平台與作業系統

| 平台 | 安裝 | 列出預設 | 執行訓練 | 備註 |
|---|---|---|---|---|
| **Linux** | `bash scripts/install_deps.sh` | `bash scripts/train_preset.sh --list` | `bash scripts/train_preset.sh gpu-12gb-4b` | 建議使用原生 CUDA |
| **macOS** | `bash scripts/install_deps.sh` | 同 Linux | `bash scripts/train_preset.sh cpu-smoke` | 無 NVIDIA CUDA；使用 CPU 冒煙或外接 GPU |
| **Windows（PowerShell）** | `powershell -File scripts/install_deps.ps1` | `powershell -File scripts/train_preset.ps1 -List` | `powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-4b` | OOM 時可設 `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` |
| **Windows（Git Bash）** | `bash scripts/install_deps.sh` | 同 Linux | 同 Linux | 命令與 Linux 相同 |

無需修改腳本即可覆寫資料集、輸出目錄或超參數：

```bash
# 自訂資料集（Linux / Git Bash）
DATASET_PATH=./data/my_dataset.jsonl bash scripts/train_preset.sh gpu-12gb-4b

# 較大資料集 — 減少 epoch（30K+ 筆）
NUM_EPOCHS=2 LEARNING_RATE=1.5e-4 DATASET_PATH=./data/large.jsonl bash scripts/train_preset.sh gpu-24gb-9b

# 自訂輸出目錄
OUTPUT_DIR=./models/my-run bash scripts/train_preset.sh gpu-12gb-4b
```

```powershell
# 自訂資料集（Windows PowerShell）
$env:DATASET_PATH = ".\data\my_dataset.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-4b

# 較大資料集 — 減少 epoch
$env:NUM_EPOCHS = "2"
$env:LEARNING_RATE = "1.5e-4"
$env:DATASET_PATH = ".\data\large.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-24gb-9b
```

### VRAM 與模型規模（QLoRA）

以下為硬體層級的起始建議，再依[資料集規模](#資料集規模指引)微調。若遇到 OOM，請先降低 `--max_seq_length`；仍不足時改用 `--target_modules q_proj,k_proj,v_proj,o_proj`。

| VRAM | 模型 | Batch | Accum | 有效批次 | Max Seq | LoRA r | Epochs | LR | 建議 GPU |
|---|---|---|---|---|---|---|---|---|---|
| 6–8 GB | Qwen3.5-4B | 1 | 16 | 16 | 2048 | 16 | 3 | 2e-4 | RTX 3060 / 4060 / 4070 |
| 12 GB | Qwen3.5-4B | 2 | 8 | 16 | 2048 | 16 | 3 | 2e-4 | RTX 3060 12 GB / 4070 |
| 12 GB | Qwen3.5-9B | 1 | 16 | 16 | 1536 | 16 | 3 | 2e-4 | RTX 5070 Ti 筆電 |
| 12 GB | Qwen3.5-9B | 1 | 16 | 16 | 2048 | 16 | 3 | 2e-4 | 僅 attention LoRA，OOM 時使用 |
| 16 GB | Qwen3.5-9B | 1 | 16 | 16 | 2048 | 32 | 3 | 2e-4 | RTX 4080 16 GB / 4070 Ti Super |
| 24 GB | Qwen3.5-4B | 2 | 8 | 16 | 4096 | 32 | 3 | 2e-4 | RTX 3090 / 4090 |
| 24 GB | Qwen3.5-9B | 2 | 8 | 16 | 2048 | 32 | 3 | 2e-4 | RTX 4090 / A5000 |
| CPU | Qwen3.5-0.8B | 1 | 4 | 4 | 512 | 8 | 1 | 2e-4 | 僅冒煙測試；需 `--device cpu` |

### 資料集規模指引

在[硬體預設](#vram-與模型規模qlora)基礎上套用下列調整。有效批次大小建議維持在 **16–32**。

| 範例數 | Epochs | 學習率 | LoRA r | test_split | 預設覆寫 | 備註 |
|---|---|---|---|---|---|---|
| < 500 | 3–5 | 1e-4 | 16 | 0.10 | `LEARNING_RATE=1e-4 NUM_EPOCHS=5` | 過擬合風險高；密切監控 eval loss |
| 500 – 5,000 | 3 | 2e-4 | 16 | 0.05 | _（預設值）_ | **內附 NTNU（~1.2K）**；小型自訂資料集的良好預設 |
| 5,000 – 30,000 | 2–3 | 1.5e-4 – 2e-4 | 32 | 0.05 | `NUM_EPOCHS=2 LEARNING_RATE=1.5e-4` | 較高 rank 有助領域適應 |
| 30,000+ | 1–2 | 1.5e-4 | 16–32 | 0.05 | `NUM_EPOCHS=2 LEARNING_RATE=1.5e-4` | 避免 3+ epoch；注意過擬合 |

**組合範例 — 12 GB 筆電 + 內附 NTNU（~1.2K）：**

```bash
bash scripts/train_preset.sh gpu-12gb-4b
# 等效手動命令 — 見[各預設的手動命令](#各預設的手動命令)
```

**組合範例 — 24 GB 桌面 + 50K 自訂範例：**

```bash
NUM_EPOCHS=2 LEARNING_RATE=1.5e-4 DATASET_PATH=./data/large.jsonl \
  bash scripts/train_preset.sh gpu-24gb-9b
```

### 具名預設組合

| 預設名稱 | 目標硬體 | 模型 | 用途 |
|---|---|---|---|
| `smoke-4b` | 任何 GPU | Qwen3.5-4B | 1 epoch 流程驗證 |
| `smoke-9b` | 12 GB+ GPU | Qwen3.5-9B | 1 epoch 流程驗證（12 GB 友善） |
| `cpu-smoke` | 僅 CPU | Qwen3.5-0.8B | 無 GPU 時驗證安裝 |
| `gpu-8gb-4b` | 6–8 GB | Qwen3.5-4B | 入門級消費級 GPU |
| `gpu-12gb-4b` | 12 GB | Qwen3.5-4B | 12 GB 上均衡的 4B 訓練 |
| `gpu-12gb-9b` | 12 GB 筆電/桌面 | Qwen3.5-9B | **建議 RTX 5070 Ti 12 GB 筆電使用** |
| `gpu-12gb-9b-long` | 12 GB | Qwen3.5-9B | 較長回答（2048 seq，僅 attention LoRA） |
| `gpu-16gb-9b` | 16 GB | Qwen3.5-9B | 全 7 層 LoRA，r=32 |
| `gpu-24gb-4b` | 24 GB | Qwen3.5-4B | 高品質 4B、長上下文 |
| `gpu-24gb-9b` | 24 GB | Qwen3.5-9B | **建議的桌面 9B 訓練** |

### 預設腳本

選用：在 Windows / Git Bash 訓練前可設定，以減輕 CUDA 記憶體碎片化：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Linux / macOS / Git Bash：**

```bash
# 列出所有預設
bash scripts/train_preset.sh --list

# 冒煙測試（正式訓練前建議先跑）
bash scripts/train_preset.sh smoke-9b

# 完整訓練 — 12 GB 筆電 + Qwen3.5-9B（內附 NTNU 資料集）
bash scripts/train_preset.sh gpu-12gb-9b

# 自訂資料集
DATASET_PATH=./data/my_dataset.jsonl bash scripts/train_preset.sh gpu-24gb-9b

# 自訂輸出目錄
OUTPUT_DIR=./models/my-run DATASET_PATH=./data/my.jsonl bash scripts/train_preset.sh gpu-12gb-4b
```

**Windows PowerShell：**

```powershell
# 列出所有預設
powershell -File scripts/train_preset.ps1 -List

# 冒煙測試
powershell -File scripts/train_preset.ps1 -Preset smoke-9b

# 完整訓練 — 12 GB 筆電
powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-9b

# 自訂資料集
$env:DATASET_PATH = ".\data\my_dataset.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-24gb-9b
```

### 各預設的手動命令

若不想使用輔助腳本，可直接複製以下命令。

**冒煙測試 — Qwen3.5-4B（任何 GPU）：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --test_split 0.05 \
  --save_steps 200 \
  --output_dir ./models/ntnu/smoke-4b
```

**12 GB — Qwen3.5-4B + NTNU（~1.2K）：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.03 \
  --test_split 0.05 \
  --logging_steps 10 \
  --save_steps 200 \
  --output_dir ./models/ntnu/gpu-12gb-4b
```

**12 GB 筆電 — Qwen3.5-9B + NTNU（~1.2K）：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 1536 \
  --num_epochs 3 \
  --per_device_batch_size 1 \
  --gradient_accum_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.03 \
  --test_split 0.05 \
  --logging_steps 10 \
  --save_steps 250 \
  --output_dir ./models/ntnu/gpu-12gb-9b
```

**24 GB 桌面 — Qwen3.5-9B + NTNU（~1.2K）：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 32 --lora_alpha 64 \
  --lora_dropout 0.05 \
  --test_split 0.05 \
  --save_steps 300 \
  --output_dir ./models/ntnu/gpu-24gb-9b
```

**24 GB 桌面 — Qwen3.5-9B + 大型自訂資料集（30K+ 筆）：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/large_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 2 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 1.5e-4 \
  --lora_r 32 --lora_alpha 64 \
  --lora_dropout 0.05 \
  --test_split 0.05 \
  --save_steps 300 \
  --output_dir ./models/custom/gpu-24gb-9b
```

**訓練後對比推論：**

```bash
python src/inference.py \
  --mode compare \
  --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --load_in_4bit True \
  --prompt "請簡單介紹國立臺灣師範大學。" \
  --no_interactive
```

---

## 資料格式

本流程支援三種輸入格式，內部皆會正規化為 ChatML 訊息格式。

### ChatML 格式

這是 Qwen3 的原生格式。每個 JSON 物件（JSONL 中每行一個，或 JSON 中作為列表元素）包含 `"messages"` 鍵：

```json
{
  "messages": [
    {"role": "system", "content": "你是一位樂於助人的助手。"},
    {"role": "user", "content": "法國的首都是哪裡？"},
    {"role": "assistant", "content": "法國的確是巴黎。"}
  ]
}
```

系統訊息（system message）為選填。此格式使用 `--dataset_format chat`（預設值）。

### Alpaca 格式

Stanford Alpaca 格式使用三個欄位：`instruction`、`input` 與 `output`。`input` 欄位可為空字串。

```json
{
  "instruction": "解釋遞迴的概念。",
  "input": "",
  "output": "遞迴是一種程式設計技巧，函式透過呼叫自身來解決相同問題的較小實例..."
}
```

```json
{
  "instruction": "將以下句子翻譯成法文。",
  "input": "你好，最近怎麼樣？",
  "output": "Bonjour, comment allez-vous ?"
}
```

此格式需使用 `--dataset_format alpaca`。

### 純文字格式

一種扁平的文字格式，每個範例的 `"text"` 欄位包含完整的對話內容。流程會以 `<|im_end|>\n<|im_start|>` 為分隔來還原使用者與助理的交替發言：

```json
{
  "text": "<|im_start|>user\n什麼是機器學習？<|im_end|>\n<|im_start|>assistant\n機器學習是人工智慧的一個子領域...<|im_end|>"
}
```

此格式需使用 `--dataset_format text`。

---

## 使用方式

### 基本訓練

使用本機 JSONL 檔案（每行一則 ChatML 格式的對話）進行訓練：

```bash
python src/train.py \
    --model_id Qwen/Qwen3.5-4B \
    --dataset_path ./data/my_dataset.jsonl \
    --output_dir ./my-finetuned-model
```

`--dataset_path` 也接受 JSONL 檔的目錄（遞迴掃描），因此可直接傳入整個 `data/` 目錄樹：

```bash
python src/train.py \
    --model_id Qwen/Qwen3.5-4B \
    --dataset_path ./data \
    --output_dir ./models/ntnu
```

### 使用 Alpaca 資料集訓練

```bash
python src/train.py \
    --dataset_path ./data/alpaca_data.json \
    --dataset_format alpaca \
    --lr 2e-4 \
    --epochs 5 \
    --output_dir ./alpaca-finetuned
```

### 從 Hugging Face Hub 載入資料集

直接從 Hugging Face Hub 載入資料集：

```bash
python src/train.py \
    --dataset_path databricks/databricks-dolly-15k \
    --dataset_format alpaca \
    --max_seq_length 2048 \
    --output_dir ./dolly-finetuned
```

### 推論 — 互動式（預設）

`src/inference.py` 預設在載入模型後進入互動式提示詞迴圈。在 `You:` 提示後輸入問題；輸入 `quit`、`exit` 或 `q`（或按 Ctrl+C）即可退出。

- **未提供 `--prompt`**：跳過批次測試，直接進入輸入迴圈。
- **有 `--prompt`**：先執行該提示一輪，再繼續互動輸入。

```bash
# 微調後模型 — 互動式對話
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/ntnu/gpu-12gb-4b

# 基底模型 — 先跑一條提示，再繼續互動
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
    --prompt "台灣師范大學的地址是什麼"

# 對比模式 — 每輪輸入同時顯示基底與微調後回應
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/ntnu/gpu-12gb-4b
```

### 推論 — 單次執行（`--no_interactive`）

適用於腳本、CI 或快速基準對比：傳入 `--no_interactive` 可執行一次批次推論後退出。未提供 `--prompt` 時，會使用兩條內建的 NTNU 測試提示。

```bash
# 基底模型基準線（微調前）
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
    --prompt "台灣師范大學的地址是什麼" --no_interactive

# 微調後模型
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./my-finetuned-model \
    --prompt "台灣師范大學的地址是什麼" --no_interactive

# 並排對比（批次模式，使用內建測試提示）
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/my-finetuned-model --no_interactive
```

### 啟用思考模式推論

若要啟用 Qwen3 的連鎖思考（chain-of-thought）推理，請傳入 `--use_thinking True`。除非同時傳入 `--no_interactive`，否則在執行 `--prompt` 後會繼續進入互動迴圈。

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./my-finetuned-model \
    --use_thinking True \
    --prompt "台灣師范大學的地址是什麼"
```

### 檢查模型

`src/inspect_model.py` 無需準備資料集即可輸出模型內部的標準化報告，適合在訓練前驗證下載的檢查點。

```bash
# 設定 + 分詞器 + 權重（預設）
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B

# 僅設定 + 分詞器（快速；略過權重下載/載入）
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --no_load

# 以 4-bit 量化載入，檢查量化後的參數量
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --load_in_4bit True
```

報告包含三個區段：

- **模型設定** — 詞表大小、隱藏（嵌入）維度、層數 / 注意力頭數 / KV 頭數、頭維度、中間（FFN）尺寸、最大位置嵌入、脈絡長度、活化函數、RoPE theta、是否綁定詞嵌入。
- **分詞器** — 分詞器類別、詞表大小、padding / truncation 方向、chat template 是否可用，以及特殊 token 及其 ID。
- **模型權重**（`--no_load` 時略過） — 模型類別、總 / 可訓練 / 凍結參數量與可訓練比例、token 嵌入形狀、嵌入記憶體估算、參數與嵌入 dtype、裝置位置。

### 檢查分詞器

`src/inspect_tokenizer.py` 顯示句子被切分為哪些 token，以及每個 token 對模型而言的樣貌——其詞表編號與 token 嵌入向量（模型所讀取的輸入列）。byte-level token 會以可讀形式顯示（`Ġ` 空格前綴轉為空格）。

```bash
# 將繁體中文句子分詞；輸出每個 token 片段、編號與嵌入向量
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "請告訴我國立台灣師範大學的具體位置在哪裡？"

python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "國立台灣師範大學的具體位置在哪裡？"

# 將英文句子分詞
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Where is the National Taiwan Normal University?"

# 查詢單一 token 字串 -> 其 index（編號）與向量
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "台"

python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "臺"

# 查詢單一 token 編號 -> 其 token 字串與向量
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token_id 120573

# 輸出完整向量（而非僅前 8 個元素）
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello" --vector_limit 0

# 僅 token 化 + 編號（快速；略過模型下載/載入）
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello world" --no_model
```

每個 token 的輸出包含 token 文字、詞表編號，以及嵌入向量（數值加上形狀、dtype 與範數 / 平均 / 標準差統計）。使用 `--vector_limit 0` 可印出向量的全部元素。

---

## 進階用法

### 自訂 LoRA 目標模組

預設情況下，LoRA 會套用至全部 7 個線性投影層：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。您可以將範圍限縮至僅注意力模組以節省記憶體：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --model_id Qwen/Qwen3.5-4B \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --output_dir ./attention-only-lora
```

### 推送至 Hugging Face Hub

若要將訓練完成的適配器直接上傳至 Hugging Face Hub，需先登入：

```bash
huggingface-cli login
```

然後加上 `--push_to_hub` 與 `--hub_model_id` 旗標：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --output_dir ./my-adapter \
    --push_to_hub True \
    --hub_model_id your-username/qwen3-4b-finetuned
```

### 完整精度 LoRA vs QLoRA

QLoRA（4-bit）為預設模式，所需記憶體最少。若要在不使用量化的情況下以 FP16 LoRA 訓練：

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --load_in_4bit False \
    --per_device_batch_size 1 \
    --output_dir ./fplora-finetuned
```

這需要約 12 GB VRAM。若遇到記憶體不足的錯誤，請降低 `per_device_batch_size` 或 `max_seq_length`。

### 訓練時啟用思考模式

如果您的資料集中包含推理過程（內容包覆在 `<think>...</think>` 中），可以啟用支援思考模式的對話模板：

```bash
python src/train.py \
    --dataset_path ./data/reasoning_dataset.jsonl \
    --use_thinking True \
    --output_dir ./thinking-finetuned
```

這樣可確保分詞器在訓練時正確處理 `<think>` 標籤。

---

## 輸出產物

訓練完成後，`output_dir` 目錄將包含以下檔案：

```
my-finetuned-model/
├── adapter_config.json         # LoRA 設定（rank、alpha、目標模組等）
├── adapter_model.safetensors   # 訓練完成的 LoRA 適配器權重
├── tokenizer_config.json       # 分詞器設定
├── tokenizer.json              # 分詞器詞彙表
├── special_tokens_map.json     # 特殊 token 定義
└── (checkpoint-XXX/)           # 訓練過程中的中間檢查點（如有設定 save_steps）
```

適配器僅約 40–80 MB，便於分享與版本管理。

---

## 建議與最佳實踐

1. **從小處著手**：正式訓練前先執行 `bash scripts/train_preset.sh smoke-4b`（9B 模型用 `smoke-9b`）。亦可手動使用 `--max_seq_length 2048` 與 `--num_epochs 1` 快速驗證。

2. **有效批次大小**：有效批次大小為 `per_device_batch_size * gradient_accum_steps * num_gpus`。建議目標值為 16–32。以 1 張 GPU 為例，`--per_device_batch_size 2 --gradient_accum_steps 8` 的有效批次大小即為 16。

3. **學習率**：對於 LoRA/QLoRA，`2e-4` 是穩健的預設值。若資料集非常小（少於 500 筆），建議降至 `1e-4` 以避免過擬合。

4. **監控評估損失**：Trainer 會自動追蹤評估損失並儲存最佳檢查點。若評估損失持續上升，可能是過擬合或學習率過高的徵兆。

5. **序列長度**：範例會透過 `SFTConfig.max_length` 截斷或填充至 `max_seq_length`。目前的訓練設定未啟用序列打包（packing）。

6. **混合精度**：在 GPU 上 `bfloat16` 的訓練穩定性優於 `float16`。若 GPU 不支援 bf16，請使用 `--torch_dtype float16`。在 CPU 上會自動改為 `float32`。

7. **資料品質重於數量**：對於指令微調而言，500–1000 筆高品質的乾淨資料集通常勝過 10,000 筆雜訊資料。

8. **推論時須對應 `model_id`**：`src/inference.py` 使用的基底模型（`--model_id`）必須與微調時的模型一致。

9. **互動式與單次推論**：手動探索時使用預設的互動迴圈；需要單次批次執行時（例如快速開始的步驟 6–7 或 shell 腳本）請加上 `--no_interactive`。

---

## 疑難排解

| 問題 | 可能原因 | 解決方案 |
|---|---|---|
| 有 GPU 但訓練仍走 CPU | 安裝了 CPU 版 PyTorch（`torch x.x.x+cpu`） | 使用 Python 3.11–3.12，再執行 `pip install -r requirements.txt` |
| `RuntimeError: GPU was requested via --device gpu` | PyTorch 無法使用 CUDA | 安裝 CUDA 版 PyTorch，或不要指定 `--device gpu` |
| `OutOfMemoryError` | 批次大小超出 GPU 記憶體 | 降低 `--per_device_batch_size` 或 `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers 版本過舊 | 升級至 `transformers>=4.47.0` |
| bitsandbytes 匯入錯誤 | 缺少 CUDA 或版本不相容 | 確認 bitsandbytes 與 CUDA 版本匹配：`pip install bitsandbytes --force-reinstall` |
| Hub 下載 / SSL 錯誤 | conda/Windows 上 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` 損壞 | `src/train.py` 與 `src/inference.py` 啟動時會自動修復；或手動修正環境變數 |
| 訓練損失出現 NaN | 學習率過高或資料類型問題 | 調低 `--learning_rate`（或 `--lr`）或改用 `--torch_dtype bfloat16` |
| 模型不斷重複相同句子 | 過擬合或溫度過低 | 提高 `--temperature` 或減少 `--num_epochs` |
| `apply_chat_template` 錯誤 | 資料格式不符合預期 | 檢查資料集是否使用正確的格式（請見[資料格式](#資料格式)） |
| 本機檔案被誤當成 Hub 資料集 | `dataset_path` 缺少 `./` 前綴 | 本機檔案請使用 `./data/train.jsonl`，勿寫成 `data/train.jsonl` |
| 評估損失持續上升 | 過擬合 | 減少 `--num_epochs`、增加 `--test_split`，或加入更多訓練資料 |
| `LoRA adapter not found` | `--adapter_path` 路徑錯誤 | 指向輸出目錄或 `checkpoint-*` 子目錄；推論會自動選取最新檢查點 |

---

## 授權條款

本專案採用 [MIT 授權條款](LICENSE)。Qwen3 / Qwen3.5 模型則依其在 Hugging Face 上的各自授權條款釋出（通常為 Apache 2.0）。

English documentation: [README.md](README.md)
