# Fine-Tuning Playground — 微調遊樂場

## 專案簡介

本專案是一個用於演示 **SFT（監督式微調，Supervised Fine-Tuning）** 的教學項目，使用 **Qwen3.5-4B** 作為基座模型。專案展示完整流程：準備資料、訓練 LoRA / QLoRA 適配器、檢查模型與分詞器、比較微調前後的回答，以及評測訓練成果。

### 什麼是 SFT？

預訓練語言模型是從大規模通用語料中學習「預測下一個 token」。它具備廣泛知識，但不一定能穩定地遵循使用者指示、採用特定語氣，或專注在某個領域。**監督式微調（SFT）** 就是在預訓練之後，用一批精心整理的「提示詞—回答」配對繼續訓練模型，讓它學會輸出我們想要形式的回答。

在本專案中，一筆訓練資料通常是包含 `user` 與 `assistant` 訊息的 ChatML 對話。對話模板會把訊息轉換成模型實際看到的文字，`trl.SFTTrainer` 再根據這段文字計算 loss。使用 `--assistant_only_loss` 時，loss 只計算助理的回答與其結束 token；`--use_thinking` 會在模板中啟用 Qwen 的 `<think>` 推理標籤。訓練完成後，可用 `src/inference.py --mode compare` 並排查看同一個問題在基座模型與微調後模型上的回答。

本演示不更新模型的全部權重，而是使用 **LoRA**：在模型的投影層加入可訓練的小矩陣，原始權重保持凍結。**QLoRA** 再搭配 4-bit NF4 量化基座權重，因此 4B 模型也能在消費級 GPU 上進行微調。

### 基座模型：Qwen3.5-4B

**Qwen3.5-4B**（[模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)）是阿里巴巴 Qwen 團隊推出的稠密（dense）、僅解碼器（decoder-only）語言模型。4B 規模適合作為微調演示：足以給出有意義的回答，也能透過 4-bit 量化在本機執行。與其他 Qwen3 / Qwen3.5 模型一樣，它支援長上下文、分組查詢注意力（GQA），並可透過對話模板中的 `<think>` 標籤選擇啟用連鎖思考的**思考模式**。

本文件的指令都以 `Qwen/Qwen3.5-4B` 作為 `--model_id` 範例。為了讓 CPU 冒煙測試快速完成，腳本預設使用 `Qwen/Qwen3.5-0.8B`（定義於 `src/model_utils.py` 的 `DEFAULT_MODEL_ID`）；訓練 4B 適配器時請明確傳入 `--model_id Qwen/Qwen3.5-4B`，並在推論與評測時使用**相同**的模型。適配器必須與訓練它的基座模型配對使用。

---

## 安裝方式

建議使用 **Python 3.11 或 3.12** 搭配相容的 PyTorch wheel。ROCm 請使用 AMD 為該版本指定的 Python 版本。每種 PyTorch 後端請使用獨立的環境。

### 方式一 — Miniconda（建議）

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

conda create -n finetune python=3.11 -y
conda activate finetune

# 安裝平台預設後端，或依下方指令選擇後端
python scripts/install_deps.py
```

### 方式二 — venv

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

# 安裝平台預設後端，或依下方指令選擇後端
python scripts/install_deps.py
```

### 安裝及驗證後端

安裝腳本會執行小型張量運算，並顯示啟用的後端與 PyTorch 版本。Apple Silicon macOS 預設使用 MPS，Intel macOS 預設使用 CPU，Windows 和 Linux 預設使用 CUDA。Bash 腳本接受與 Python 安裝腳本相同的旗標。Windows PowerShell 腳本使用 `-Backend` 和 `-Rocm4Bit`。

| 後端 | 在目前環境安裝 | 驗證模型工具 |
|---|---|---|
| NVIDIA CUDA（Linux / Windows） | `python scripts/install_deps.py --backend cuda` | `python src/inspect_model.py --device cuda --no_load` |
| AMD ROCm（Linux / WSL2 / Windows） | 先安裝 AMD PyTorch wheel，再執行 `python scripts/install_deps.py --backend rocm` | `python src/inspect_model.py --device rocm --no_load` |
| Apple M 系列（macOS） | 使用原生 arm64 Python 執行 `bash scripts/install_deps.sh --backend mps` | `python src/inspect_model.py --device mps --no_load` |
| Intel XPU（Linux / Windows） | 安裝 GPU 驅動程式後執行 `python scripts/install_deps.py --backend xpu` | `python src/inspect_model.py --device xpu --no_load` |
| CPU | `python scripts/install_deps.py --backend cpu` | `python src/inspect_model.py --device cpu --no_load` |

使用 ROCm 時，請先確認驅動程式、GPU 與 PyTorch wheel 相容。安裝腳本會沿用環境中的 AMD wheel。使用 Intel GPU 前請先確認 GPU 和驅動程式的需求。 [AMD Linux 安裝](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) · [AMD WSL2 相容性](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html) · [AMD Windows 安裝](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html) · [Intel XPU 安裝](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)

若要使用 AMD NF4，請執行 `python scripts/install_deps.py --backend rocm --rocm-4bit` 或 `powershell -File scripts/install_deps.ps1 -Backend rocm -Rocm4Bit`。請確認 bitsandbytes 套件與 GPU 和 ROCm 版本相容。 [bitsandbytes 安裝指南](https://huggingface.co/docs/bitsandbytes/installation#amd-rocm)

NVIDIA CUDA 仍可直接執行 `python -m pip install -r requirements.txt` 安裝。使用其他後端時，安裝腳本選定或確認 PyTorch 後會使用 `requirements-common.txt`。

核心套件：

| 套件 | 最低版本 | 用途 |
|---|---|---|
| `torch` | >= 2.4.0, < 2.12 | 深度學習框架（`requirements.txt` 預設 CUDA cu128） |
| `transformers` | >= 5.12.1 | Qwen 模型載入、分詞與訓練 |
| `peft` | >= 0.13.0 | LoRA / QLoRA 適配器設定 |
| `trl` | >= 1.6.0 | 支援助理標籤和分塊 loss 的 `SFTTrainer` |
| `accelerate` | >= 1.13.0 | Trainer 裝置配置和混合精度 |
| `bitsandbytes` | CUDA >= 0.46.1、ROCm >= 0.50.0 | 選用的 4-bit 量化（QLoRA） |
| `datasets` | >= 3.0.0 | 資料集載入與處理 |
| `packaging` | >= 21.0 | PyTorch 版本檢查 |
| `certifi` | >= 2023.7.22 | 修復 Hub 下載所用的 CA bundle |

---

## 快速開始

### 跨平台冒煙測試

安裝後端後，使用 Qwen3.5-0.8B（腳本預設模型）訓練一步並重新載入適配器。可將 `gpu` 改為 `mps`、`rocm`、`xpu`、`cuda` 或 `cpu` 來指定裝置。

```bash
python -c "from pathlib import Path; p=Path('models/backend-smoke'); p.mkdir(parents=True, exist_ok=True); (p/'train.jsonl').write_bytes(Path('data/ntnu_dataset.jsonl').read_bytes().splitlines(keepends=True)[0])"
python src/train.py --device gpu --dataset_path ./models/backend-smoke/train.jsonl --num_epochs 1 --test_split 0 --max_seq_length 64 --per_device_batch_size 1 --gradient_accum_steps 1 --output_dir ./models/backend-smoke/adapter
python src/inference.py --device gpu --mode finetuned --adapter_path ./models/backend-smoke/adapter --prompt "分詞器的用途是什麼？" --max_new_tokens 32 --no_interactive
```

使用相容的 AMD GPU 時，可在兩道指令都加上 `--load_in_4bit True` 來驗證 NF4。

### NVIDIA CUDA 上的 Qwen3.5-4B

使用內附的臺師大資料集（1,252 筆 ChatML 範例）訓練，再比較基座模型與適配器：

```bash
# 訓練
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device cuda \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-12gb-4b

# 微調前
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# 微調後
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# 可選：互動式對話，或並排比較
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "請簡單介紹國立臺灣師範大學。" --no_interactive
```

更多資料格式、訓練選項與推論模式請見[使用方式](#使用方式)。

---

## 專案結構

```
fine-tuning-playground/
├── requirements.txt          # NVIDIA cu128 安裝入口
├── requirements-common.txt   # 共用 Python 套件依賴
├── LICENSE                   # MIT 授權條款
├── src/                      # 原始碼腳本
│   ├── train.py              # 完整 SFT 流程（單檔、由上而下）
│   ├── inference.py          # 互動式推論 + 基底/微調/對比模式
│   ├── evaluate_ntnu.py      # 貪婪解碼與臺師大品質盲評
│   ├── model_utils.py        # 共用的分詞器、模型載入與 SSL 工具
│   ├── device_utils.py       # 後端選擇、精度與 ROCm NF4 檢查
│   ├── inspect_model.py      # 模型內部檢查（參數量、詞表、維度、設定）
│   └── inspect_tokenizer.py  # 分詞器檢查（token 片段、編號、嵌入向量）
├── data/                     # 內附訓練資料集
│   ├── ntnu_dataset.jsonl    # 1,252 筆臺師大 ChatML 範例
│   ├── full_dataset.jsonl    # 1,746 筆臺師大 + 臺灣彙整資料
│   ├── ntnu/                 # 臺師大分類 JSONL 檔
│   └── taiwan/               # 臺灣分類 JSONL 檔
├── evaluation/ntnu_qa.jsonl  # 60 題獨立測試題與評分準則（事實題附來源）
├── models/                   # LoRA 適配器與評測輸出（Git 忽略）
├── scripts/                  # 依賴安裝腳本
│   ├── install_deps.py       # 選擇後端、安裝依賴與驗證裝置
│   ├── install_deps.sh       # Bash 包裝器
│   └── install_deps.ps1      # PowerShell 包裝器
├── tests/
│   └── test_device_utils.py  # 後端選擇和量化測試
├── README.md                 # 英文文件
└── README.zh-Hant.md         # 本文件（繁體中文）
```

### 各檔案說明

**`src/train.py`** — 完整的監督式微調流程，全部寫在一個檔案中。依序執行：SSL 修復 → 解析命令列參數 → 選擇裝置 → 載入分詞器 → 載入基座模型（可選 4-bit）→ 套用 LoRA → 載入並格式化資料集 → 設定 `SFTTrainer` → 訓練 → 儲存適配器。所有超參數皆為 CLI 旗標，無獨立設定模組。

**`src/inference.py`** — 獨立推論腳本，支援基座模型、微調後模型或並排對比。預設在載入模型後進入互動式迴圈（`You:` 提示；輸入 `quit`、`exit` 或 `q` 退出）。CLI 旗標：`--mode`（`base` / `finetuned` / `compare`）、`--model_id`、`--adapter_path`、`--prompt`、`--load_in_4bit`、`--torch_dtype`、`--device`、`--max_new_tokens`、`--temperature`、`--top_p`、`--use_thinking`、`--no_interactive`。

**`src/evaluate_ntnu.py`** — 以 `generate` 生成固定題組的確定性回答，以 `blind` 將多組回答洗牌供盲評，並以 `score` 套用品質門檻。**`evaluation/ntnu_qa.jsonl`** 內含 60 題獨立試題：40 題事實題（每題附官方來源）、10 題不確定資訊題及 10 題一般題。請勿將這些題目放入訓練資料。

**`src/model_utils.py`** — 共用的 SSL 憑證修復、分詞器載入、NF4 設定與基座模型載入工具。`DEFAULT_MODEL_ID` 為 `Qwen/Qwen3.5-0.8B`，供冒煙測試使用。

**`src/device_utils.py`** — 為各模型指令選擇後端、有效精度和 NF4 設定，包含在 ROCm 上啟用 4-bit 前的核心驗證。

**`src/inspect_model.py`** — 輸出模型內部的標準化報告：總/可訓練/凍結參數量、token 嵌入形狀與記憶體估算、詞表大小、層級結構、注意力與 FFN 設定、RoPE 參數及分詞器中繼資料。CLI 旗標：`--model_id`、`--load_in_4bit`、`--torch_dtype`、`--device`、`--no_load`（僅輸出設定與分詞器）。

**`src/inspect_tokenizer.py`** — 輸出句子被切分為哪些 token、每個 token 的詞表編號，以及從模型輸入嵌入層取出的向量。也支援以 `--token`（字串）或 `--token_id`（編號）查詢單一 token。byte-level token 會以可讀形式顯示。CLI 旗標：`--model_id`、`--sentence`、`--token`、`--token_id`、`--vector_limit`（`0` = 全部元素）、`--load_in_4bit`、`--torch_dtype`、`--device`、`--no_model`（僅輸出 token 化與編號）。

**`data/`** — 包含 `ntnu_dataset.jsonl`（1,252 筆關於國立臺灣師範大學的 ChatML 紀錄）、`full_dataset.jsonl`（1,746 筆，涵蓋臺師大、臺灣的大學、縣市、歷史與文化），以及 `ntnu/` 與 `taiwan/` 下的分類紀錄。

**`scripts/install_deps.py`** — 安裝並驗證選擇的 PyTorch 後端與共用依賴；`install_deps.sh` 和 `install_deps.ps1` 只是將參數轉送給它。**`tests/test_device_utils.py`** 以模擬硬體測試後端與精度選擇；以 `python -m unittest discover -s tests -v` 執行。

---

## 設定參數

### 訓練（`src/train.py`）

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face 模型識別碼（演示時使用 `Qwen/Qwen3.5-4B`） |
| | `load_in_4bit` | CUDA 預設開啟 | NF4 QLoRA。相容的 ROCm 可設 `True`；MPS / XPU 不支援；CPU 使用未量化權重 |
| | `torch_dtype` | `bfloat16` | 計算精度（CPU 固定 `float32`；不支援 BF16 的 GPU 改用 FP16） |
| **LoRA** | `lora_r` | `16` | LoRA 秩（rank） |
| | `lora_alpha` | `32` | LoRA 縮放因子（通常設為 2 × r） |
| | `lora_dropout` | `0.05` | LoRA 層的 dropout 比率 |
| | `target_modules` | 全部 7 個線性層 | 逗號分隔的模組名稱（預設 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`） |
| **資料** | `dataset_path` | （必填） | 本機 JSON/JSONL 檔案、JSONL 檔目錄（遞迴掃描），或 HF 資料集名稱。已存在的本機路徑會自動判定 |
| | `dataset_format` | `chat` | 輸入格式：`chat`、`alpaca` 或 `text` |
| | `test_split` | `0.05` | 保留為評估集的比例（0 = 不評估） |
| | `seed` | `42` | 資料切分、適配器初始化和取樣的亂數種子 |
| | `max_seq_length` | `4096` | 每個範例的最大 token 數 |
| | `use_thinking` | `False` | 在對話模板中啟用思考模式 |
| **訓練** | `learning_rate`（`--lr`） | `2e-4` | 峰值學習率 |
| | `num_epochs`（`--epochs`） | `3` | 訓練週期數 |
| | `per_device_batch_size`（`--batch_size`） | `2` | 每張 GPU 的批次大小 |
| | `gradient_accum_steps` | `8` | 梯度累積步數 |
| | `gradient_checkpointing` | `True` | 重算 activation，降低 GPU 記憶體峰值 |
| | `group_by_length` | 關閉 | 將長度相近的訓練範例分在同一批次，減少 padding |
| | `assistant_only_loss` | 關閉 | 在 ChatML 資料中只監督助理回答及結束 token；目標被截斷時會停止 |
| | `loss_type` | TRL 預設 | 可選 `nll` 或 `chunked_nll`，分塊可降低詞表投影的記憶體需求 |
| | `eval_each_epoch` | 關閉 | 每輪評估並儲存，最後載入驗證 loss 最低的檢查點 |
| | `eval_batch_size` | `8` | 每張 GPU 的評估批次大小 |
| | `warmup_ratio` | `0.03` | 線性 warmup 比例 |
| | `logging_steps` | `10` | 每隔 N 步記錄一次指標 |
| | `save_steps` | `200` | 每隔 N 步儲存一次檢查點 |
| | `output_dir` | `./models/qwen3-4b-finetuned` | 輸出目錄 |
| **基準測試** | `benchmark_steps` | `0` | 要計時的 optimizer 步數。大於零時跳過驗證和匯出適配器 |
| | `benchmark_warmup_steps` | `3` | 計時前排除的暖機步數 |
| | `run_metrics` | — | 選用的 JSON 紀錄，包含耗時、實際後端、dtype、NF4 狀態、序列長度、模型 revision 和套件版本 |
| **硬體** | `device` | `auto` | `auto`、`gpu`、`cuda`、`rocm`、`mps`、`xpu`、`cpu` |
| **Hub** | `push_to_hub` | `False` | 將適配器推送至 HF Hub |
| | `hub_model_id` | `""` | Hub 上的目標儲存庫名稱 |

### 推論（`src/inference.py`）

| 類別 | 參數 | 預設值 | 說明 |
|---|---|---|---|
| **模型** | `model_id` | `Qwen/Qwen3.5-0.8B` | 基底模型（須與微調時使用的模型一致） |
| | `mode` | `base` | `base`、`finetuned` 或 `compare` |
| | `adapter_path` | — | LoRA 適配器目錄（`finetuned` / `compare` 模式必填） |
| | `load_in_4bit` | CUDA 預設開啟 | NF4 推論。相容的 ROCm 可設 `True`；MPS / XPU 不支援；CPU 使用未量化權重 |
| | `torch_dtype` | `bfloat16` | 計算精度 |
| | `device` | `auto` | `auto`、`gpu`、`cuda`、`rocm`、`mps`、`xpu`、`cpu` |
| **生成** | `prompt` | — | 單次測試提示；搭配 `--no_interactive` 時**必填** |
| | `max_new_tokens` | `2048` | 最大生成 token 數 |
| | `temperature` | `0.7` | 取樣溫度 |
| | `top_p` | `0.9` | 核取樣（nucleus sampling）閾值 |
| | `use_thinking` | `False` | 在對話模板中啟用思考模式 |
| | `no_interactive` | `False` | 執行一次批次推論後退出，不進入互動迴圈 |

### 檢查（`src/inspect_model.py`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-0.8B` | 要檢查的 Hugging Face 模型識別碼 |
| `load_in_4bit` | `False` | 以 4-bit 量化載入權重（查看量化後的參數量） |
| `torch_dtype` | `float32` | 載入權重時使用的計算精度 |
| `device` | `auto` | `auto`、`gpu`、`cuda`、`rocm`、`mps`、`xpu`、`cpu` |
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
| `device` | `auto` | `auto`、`gpu`、`cuda`、`rocm`、`mps`、`xpu`、`cpu` |
| `no_model` | `False` | 僅輸出 token 化與編號，不載入模型（無向量） |

`--sentence`、`--token`、`--token_id` 三者至少需提供其一。

> **備註：** `src/train.py` 與 `src/inference.py` 的 `--model_id` 預設皆為 `Qwen/Qwen3.5-0.8B`（適合 CPU 冒煙測試）。演示時請傳入 `--model_id Qwen/Qwen3.5-4B`，推論與評測也須使用**相同**的 `--model_id`。在 `src/inference.py` 中，省略 `--prompt` 會跳過批次推論並直接進入互動迴圈；使用 `--no_interactive` 時則必須提供非空的 `--prompt`。

### 臺師大評測（`src/evaluate_ntnu.py generate`）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--cases`（在全域、子命令之前） | `evaluation/ntnu_qa.jsonl` | 相容的 JSONL 題組檔 |
| `model_id` | （必填） | 適配器所使用的基座模型 |
| `adapter_path` | （必填） | 適配器目錄（或包含 `checkpoint-*` 的上層目錄） |
| `device` | `gpu` | `auto`、`gpu`、`cuda`、`rocm`、`mps`、`xpu`、`cpu` |
| `load_in_4bit` | CUDA 預設開啟 | 用 `True` 或 `False` 指定 NF4。ROCm 需設為 `True` 才啟用 |
| `bf16_weights` | 關閉 | 在支援時使用未量化的 BF16 權重。與 `--load_in_4bit` 互斥 |
| `batch_size` | `4` | 生成 batch 大小 |
| `max_new_tokens` | `192` | 回答長度上限 |
| `output` | （必填） | 存放可續跑回答與執行中繼資料的 JSON 檔 |

生成採用貪婪解碼，並將實際後端、dtype 和量化方式寫入輸出檔供續跑比對。切換硬體後端或精度時請指定新的輸出路徑。`blind` 和 `score` 的步驟請見[臺師大評測流程](#臺師大評測流程)。

---

## 訓練建議

本專案使用 `bfloat16`、ChatML 資料（`--dataset_format chat`，預設值），以及 `per_device_batch_size × gradient_accum_steps` 的有效批次大小。預設 `test_split 0.05` 時，Trainer 會在獨立的驗證集上評估，並在訓練結束時載入 eval loss 最低的檢查點。若記憶體不足，請先降低 `--max_seq_length`。

### 資料集規模指引

| 範例數 | Epochs | 學習率 | test_split | 備註 |
|---|---|---|---|---|
| < 500 | 3–5 | 1e-4 | 0.10 | 過擬合風險高；密切監控 eval loss |
| 500 – 5,000 | 3 | 2e-4 | 0.05 | 內附臺師大資料集（1,252 筆）；小型自訂資料集的良好預設 |
| 5,000 – 30,000 | 2–3 | 1.5e-4 – 2e-4 | 0.05 | 較高 LoRA rank 有助領域適應 |
| 30,000+ | 1–2 | 1.5e-4 | 0.05 | 避免 3+ epoch；注意過擬合 |

### 範例指令

下列指令直接呼叫 `src/train.py`，不需要任何預設腳本。

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
  --output_dir ./models/ntnu/smoke-4b
```

**12 GB GPU — Qwen3.5-9B：**

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
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-12gb-9b
```

**24 GB GPU — Qwen3.5-9B 搭配較大資料集：**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/full_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 2 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 1.5e-4 \
  --lora_r 32 --lora_alpha 64 \
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-24gb-9b
```

---

## 臺師大評測流程

固定題組共 60 題：40 題事實題（每題附官方來源）、10 題不確定資訊題及 10 題一般題。請勿將這些題目放入訓練資料。

先為每個要比較的適配器生成回答，再準備盲評。`blind` 接受不重複的 `NAME=PATH` 參數，並另存一份對照 key；`score` 需要恰好兩個基線名稱。

```bash
python src/evaluate_ntnu.py generate \
  --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --output ./models/ntnu/eval/candidate.json

python src/evaluate_ntnu.py blind \
  "existing=./models/ntnu/eval/existing.json" \
  "baseline=./models/ntnu/eval/baseline.json" \
  "candidate=./models/ntnu/eval/candidate.json" \
  --output ./models/ntnu/eval/review.json \
  --key ./models/ntnu/eval/key.json

python src/evaluate_ntnu.py score \
  --review ./models/ntnu/eval/review.json \
  --key ./models/ntnu/eval/key.json \
  --baselines existing baseline \
  --output ./models/ntnu/eval/quality.json
```

請對照每條準則評分。若捏造有實質影響的事實，例如資格、截止日、金額、地址或操作指示，將 `major` 標為真。答案符合所有準則，且事實題使用繁體中文，才將 `pass` 標為真；請填寫兩項布林值與失分原因。`score` 只有在 fact ≥ `max(34, 兩個基線)`、uncertain ≥ 8、general ≥ `最佳基線 − 1`，且 `major == 0` 時才會給候選 `passes_quality`。此外，可比較 `--run_metrics` 產生的 `elapsed_seconds`；為使耗時可比較，一次只執行一個訓練或推論工作。

---

## 資料格式

本流程支援三種輸入格式，訓練前皆會轉換為對話表示。

### ChatML 格式

這是 Qwen3 / Qwen3.5 的原生格式。每個 JSON 物件（JSONL 中每行一個，或 JSON 中作為列表元素）包含 `messages` 鍵：

```json
{
  "messages": [
    {"role": "system", "content": "你是一位樂於助人的助手。"},
    {"role": "user", "content": "法國的首都是哪裡？"},
    {"role": "assistant", "content": "法國的首都是巴黎。"}
  ]
}
```

系統訊息為選填。此格式使用 `--dataset_format chat`（預設值）。

### Alpaca 格式

Stanford Alpaca 格式使用三個欄位：`instruction`、`input` 與 `output`。`input` 欄位可為空字串。

```json
{
  "instruction": "解釋遞迴的概念。",
  "input": "",
  "output": "遞迴是一種程式設計技巧，函式透過呼叫自身來解決相同問題的較小實例..."
}
```

此格式使用 `--dataset_format alpaca`。

### 純文字格式

一種扁平的文字格式，每個範例的 `text` 欄位包含完整對話內容。流程會以 `<|im_end|>\n<|im_start|>` 為分隔來還原使用者與助理的交替發言：

```json
{
  "text": "<|im_start|>user\n什麼是機器學習？<|im_end|>\n<|im_start|>assistant\n機器學習是人工智慧的一個子領域...<|im_end|>"
}
```

此格式使用 `--dataset_format text`。

---

## 使用方式

### 基本訓練

使用本機 JSONL 檔案（每行一則 ChatML 格式的對話）進行訓練：

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/my_dataset.jsonl \
  --output_dir ./models/my-adapter
```

`--dataset_path` 也接受 JSONL 檔的目錄（遞迴掃描），因此可直接傳入整個 `data/` 目錄樹。已存在的本機路徑會自動判定；若路徑在本機不存在，則會被視為 Hugging Face 資料集名稱。

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
  --output_dir ./models/alpaca-finetuned
```

### 從 Hugging Face Hub 載入資料集

請使用欄位符合所選格式的 Hub 資料集，例如 `tatsu-lab/alpaca` 具有 `instruction`、`input`、`output` 三個欄位：

```bash
python src/train.py \
  --dataset_path tatsu-lab/alpaca \
  --dataset_format alpaca \
  --max_seq_length 2048 \
  --output_dir ./models/alpaca-hub-finetuned
```

### 推論 — 互動式（預設）

`src/inference.py` 預設在載入模型後進入互動式提示詞迴圈。在 `You:` 提示後輸入問題；輸入 `quit`、`exit` 或 `q`（或按 Ctrl+C）即可退出。

- **未提供 `--prompt`**：跳過批次推論，直接進入輸入迴圈。
- **有 `--prompt`**：先執行該提示一次，再繼續互動輸入。

```bash
# 微調後模型 — 互動式對話
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b

# 基底模型 — 先跑一條提示，再繼續互動
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼"

# 對比模式 — 每輪輸入同時顯示基底與微調後回應
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b
```

### 推論 — 單次執行（`--no_interactive`）

適用於腳本或 CI：傳入 `--no_interactive` 可執行一次提示後退出。**必須提供非空的 `--prompt`**，否則腳本會報錯。

```bash
# 基底模型基準線（微調前）
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# 微調後模型
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/my-adapter \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# 並排對比
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "請簡單介紹國立臺灣師範大學。" --no_interactive
```

### 啟用思考模式推論

若要啟用 Qwen3 / Qwen3.5 的連鎖思考推理，請傳入 `--use_thinking True`。除非同時傳入 `--no_interactive`，否則在執行 `--prompt` 後會繼續進入互動迴圈。

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/my-adapter \
  --use_thinking True \
  --prompt "台灣師範大學的地址是什麼"
```

### 檢查模型

`src/inspect_model.py` 無需準備資料集即可輸出模型內部的標準化報告。

```bash
# 設定 + 分詞器 + 權重（預設）
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B

# 僅設定 + 分詞器（略過權重載入）
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --no_load

# 以 4-bit 量化載入，檢查量化後的參數量
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --load_in_4bit True
```

報告涵蓋模型設定（詞表大小、嵌入維度、層數、注意力頭數、KV 頭數、頭維度、FFN 尺寸、脈絡長度、活化函數、RoPE theta、是否綁定詞嵌入）、分詞器中繼資料，以及模型權重（參數量、嵌入形狀與記憶體估算、dtype、裝置位置）。

### 檢查分詞器

`src/inspect_tokenizer.py` 顯示句子被切分為哪些 token，以及每個 token 的詞表編號與嵌入向量。byte-level token 會以可讀形式顯示。

```bash
# 將句子分詞；輸出每個 token 片段、編號與嵌入向量
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Where is National Taiwan Normal University?"

# 查詢單一 token 字串或編號
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "台"
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token_id 120573

# 輸出完整向量（而非僅前 8 個元素）
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello" --vector_limit 0

# 僅 token 化 + 編號（略過模型載入，因此沒有向量）
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello world" --no_model
```

---

## 進階用法

### 自訂 LoRA 目標模組

預設情況下，LoRA 會套用至全部 7 個線性投影層：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。您可以將範圍限縮至注意力模組：

```bash
python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --model_id Qwen/Qwen3.5-4B \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --output_dir ./models/attention-only-lora
```

### 推送至 Hugging Face Hub

先登入，再傳入 `--push_to_hub True` 與目標儲存庫：

```bash
huggingface-cli login

python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --output_dir ./models/my-adapter \
  --push_to_hub True \
  --hub_model_id your-username/qwen3-4b-finetuned
```

### 完整精度 LoRA vs QLoRA

QLoRA（4-bit）是 NVIDIA 的預設模式。若要使用未量化的 LoRA，請指定 `--load_in_4bit False`：

```bash
python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --load_in_4bit False \
  --per_device_batch_size 1 \
  --output_dir ./models/lora-finetuned
```

未量化的 4B 訓練所需記憶體遠高於 4-bit 訓練。若遇到記憶體不足，請降低 `per_device_batch_size` 或 `max_seq_length`。

### 訓練時啟用思考模式

如果您的資料集包含包覆在 `<think>...</think>` 中的推理過程，可啟用支援思考模式的對話模板：

```bash
python src/train.py \
  --dataset_path ./data/reasoning_dataset.jsonl \
  --use_thinking True \
  --output_dir ./models/thinking-finetuned
```

---

## 輸出產物

訓練完成後，`output_dir` 目錄將包含以下檔案：

```
my-adapter/
├── adapter_config.json         # LoRA 設定（rank、alpha、目標模組等）
├── adapter_model.safetensors   # 訓練完成的 LoRA 適配器權重
├── tokenizer_config.json       # 分詞器設定
├── tokenizer.json              # 分詞器詞彙表
├── special_tokens_map.json     # 特殊 token 定義
└── checkpoint-*/               # 訓練過程中的中間檢查點（有達到 save_steps 時）
```

只會儲存體積很小的適配器，基座模型不會被複製到 `output_dir`。基準測試（`--benchmark_steps > 0`）只記錄指標，不會匯出適配器。

---

## 建議與最佳實踐

1. **從小處著手**：先執行[快速開始](#快速開始)中的單筆冒煙測試，或使用 `--max_seq_length 2048 --num_epochs 1` 快速驗證。

2. **有效批次大小**：`per_device_batch_size × gradient_accum_steps × num_gpus`。建議目標值為 16–32；以 1 張 GPU 為例，`--per_device_batch_size 2 --gradient_accum_steps 8` 即為 16。

3. **學習率**：對於 LoRA/QLoRA，`2e-4` 是穩健的預設值。若資料少於 500 筆，可試試 `1e-4`。

4. **監控評估損失**：當 `--test_split` 大於 0 時，Trainer 會追蹤 eval loss 並在最後載入最佳檢查點。eval loss 上升代表可能過擬合或學習率過高。

5. **序列長度**：範例會透過 `SFTConfig.max_length` 截斷或填充；目前未啟用序列打包（packing）。

6. **混合精度**：啟動紀錄會顯示有效 dtype。不支援 BF16 的 GPU 會改用 FP16，CPU 使用 FP32；MPS 訓練使用 FP16 需 PyTorch 2.5+，BF16 需 2.6+。

7. **資料品質重於數量**：對於指令微調，500–1000 筆乾淨資料通常勝過 10,000 筆雜訊資料。

8. **推論時須對應 `model_id`**：`--model_id` 必須與微調時使用的基座模型一致。

9. **互動式與單次推論**：手動探索時使用預設的互動迴圈；腳本與 CI 請同時指定 `--no_interactive` 與 `--prompt`。

---

## 疑難排解

| 問題 | 可能原因 | 解決方案 |
|---|---|---|
| 有 GPU 但訓練仍走 CPU | 目前的 PyTorch wheel 或驅動無法使用 GPU | 在新環境依[安裝指引](#安裝及驗證後端)設定，再明確指定 `--device` |
| `--device gpu` 或指定的 GPU 後端不可用 | PyTorch 無法使用所選加速器 | 確認該後端的 wheel 和驅動。要用 CPU 執行可指定 `--device cpu` |
| `OutOfMemoryError` | 批次大小或序列長度超出記憶體 | 降低 `--per_device_batch_size` 或 `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers 版本過舊 | 升級至 `transformers>=5.12.1` |
| CUDA 或 ROCm 的 NF4 失敗 | 缺少 bitsandbytes 或 GPU 核心不相容 | 確認 bitsandbytes 與 GPU 及 CUDA/ROCm 版本匹配。使用 `--load_in_4bit False` 可載入未量化權重 |
| MPS 或 XPU 拒絕 4-bit 載入 | 這些後端不支援 NF4 | 改用 `--load_in_4bit False` 重新執行 |
| MPS 運算子不受支援 | PyTorch 缺少該運算子的 MPS 實作 | 啟動 Python 前，先設置 `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Hub 下載 / SSL 錯誤 | `SSL_CERT_FILE`、`REQUESTS_CA_BUNDLE` 或 `CURL_CA_BUNDLE` 損壞 | 訓練、推論與檢查工具啟動時會修復這些路徑；或手動修正環境變數 |
| 訓練損失出現 NaN | 學習率過高或資料類型問題 | 調低 `--learning_rate`（或 `--lr`）或改用 `--torch_dtype bfloat16` |
| 模型不斷重複相同句子 | 過擬合或溫度過低 | 提高 `--temperature` 或減少 `--num_epochs` |
| `apply_chat_template` 錯誤 | 資料格式不符合預期 | 檢查資料集是否符合[資料格式](#資料格式) |
| 本機檔案被誤當成 Hub 資料集 | 路徑在本機不存在，因此被當成資料集名稱 | 檢查路徑；使用 `./data/train.jsonl` 可明確指向本機檔案 |
| `--no_interactive` 發生參數錯誤 | 未提供 `--prompt` | 使用 `--no_interactive` 時請提供非空的 `--prompt` |
| 評估損失持續上升 | 過擬合 | 減少 `--num_epochs`、增加 `--test_split`，或加入更多訓練資料 |
| `LoRA adapter not found` | `--adapter_path` 路徑錯誤 | 指向輸出目錄或包含 `checkpoint-*` 的上層目錄；推論會自動選取最新檢查點 |

---

## 授權條款

本專案採用 [MIT 授權條款](LICENSE)。Qwen3 / Qwen3.5 模型則依其在 Hugging Face 上的各自授權條款釋出（通常為 Apache 2.0）。

English documentation: [README.md](README.md)
