# LLM-Based Network Management: Natural-Language-to-CLI Command Generation

**Mistral-7B-Instruct-v0.2 · LoRA / 4-bit MLX · Network management**

This repository presents a proof-of-concept that translates **natural-language network-management requests into structured CLI commands**. It uses a 4-bit, MLX-compatible Mistral-7B-Instruct-v0.2 model with a saved LoRA adapter and a constrained command specification.

The scope is deliberately narrow: **it generates commands; it does not execute them**, perform network operations, or provide a trained intent classifier. This is a small, task-specific prototype rather than a production network-management agent.

## What the project does

Given a request such as *“Show me the network graph”* or *“Export the configuration of this element to a directory”*, the model proposes a command following the `br-t9s.cli extract ...` grammar. The supported extraction families are:

| Operation | Command shape |
| --- | --- |
| Element configuration | `br-t9s.cli extract config <element> <directory>` |
| Network graph | `br-t9s.cli extract graph` |
| Infrastructure | `br-t9s.cli extract infra` |
| Element logs | `br-t9s.cli extract logs <element>` |

The prompt also includes variants with flags such as `--config`, `--namespace`, and `--verbose`.

### Recorded command-generation examples

The following examples are preserved from the **original development notebook**; they have not been freshly reproduced for this release.

| Natural-language request (abbreviated) | Recorded command |
| --- | --- |
| Troubleshoot the network diagram | `br-t9s.cli extract graph --verbose` |
| Retrieve the configuration of a specified network element to `/shared_data/` | `br-t9s.cli extract config oai-amf.oai-cn-rz2zjpq8 /shared_data/` |
| Export the `gnb` configuration as YAML | `br-t9s.cli extract config gnb yaml/config.yaml` |

These demonstrations show the output format only. The generated commands have **not** been validated against a live network or used as an end-to-end accuracy benchmark.

## Method

The notebook provides a detailed prompt specifying the allowed CLI syntax and the information to extract from each user request. The fine-tuning dataset contains paired examples of natural-language requests and target commands formatted using Mistral's `[INST] ... [/INST]` instruction tokens.

The original fine-tuning setup used LoRA on the model's query (`q_proj`) and value (`v_proj`) projections in the **last 16 transformer layers**, with **rank 4**. The supplied `adapters.npz` contains 64 parameter arrays (approximately **852 thousand LoRA parameters**). The base Mistral-7B model is not included because the adapter is loaded separately.

## Repository contents

```text
llm-network-command-generation/
├── README.md
├── network_cli_generation.ipynb
├── requirements.txt
├── adapters.npz
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── scripts/
│   ├── lora.py
│   ├── models.py
│   └── utils.py
└── data/
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

The three `scripts/` files form the **original legacy MLX runtime** for loading the model, configuring LoRA, running fine-tuning, and generating text. They are included together because `lora.py` depends on the local `models.py` and `utils.py` modules. The unfinished classifier (`lora2.py`) and unrelated development notebooks are intentionally excluded.

## Dataset

The included JSONL data consists of **98 command-generation examples**:

| File | Examples | Purpose |
| --- | ---: | --- |
| [`data/train.jsonl`](data/train.jsonl) | 78 | Fine-tuning |
| [`data/valid.jsonl`](data/valid.jsonl) | 10 | Validation |
| [`data/test.jsonl`](data/test.jsonl) | 10 | Held-out evaluation data |

Each nonblank JSONL line is an object containing a `text` field that holds both the instruction and the expected command. The examples cover configuration, graph, infrastructure, and log extraction.

**Dataset limitations:** The data is small and contains many paraphrases of requests for commands from a predefined grammar. All commands appearing as validation/test targets are also present among the training targets, even though the full text records do not overlap across splits. This is a proof-of-concept dataset; the project does not establish generalization to unfamiliar commands, networks, or CLI grammars. Confirm that any included example identifiers and paths are appropriate to publish.

## Setup (Apple Silicon / macOS)

The original project used **Apple's MLX framework** and a **legacy MLX LoRA example**. An Apple Silicon Mac and compatible MLX installation are required to use the bundled code. The exact historical package versions were not preserved, so current-package compatibility may require adjustments.

1. Clone or download this repository and create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Make sure a compatible copy of the base model is available. By default, the notebook and training commands point to [`mlx-community/Mistral-7B-Instruct-v0.2-4bit`](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit). The legacy `utils.py` can attempt to download it if it is not cached. Alternatively, replace the model path with a compatible local MLX model directory.

3. Start Jupyter **from the repository root**:

   ```bash
   jupyter lab
   ```

4. Open [`network_cli_generation.ipynb`](network_cli_generation.ipynb), review `MODEL_PATH` and `LORA_SCRIPT`, and run the inference cells. The default notebook configuration is:

   ```python
   MODEL_PATH = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
   ADAPTER_PATH = Path("adapters.npz")
   LORA_SCRIPT = Path("scripts/lora.py")
   ```

**Important:** This checkpoint is stored as a historical `.npz` adapter, not the newer adapter-directory format supported by some `mlx-lm` interfaces. This repository supplies the corresponding old `lora.py`, `models.py`, and `utils.py` rather than relying on a current `mlx-lm` API.

### Fine-tuning using the supplied dataset

The original development notebook recorded the following training configuration: **100 iterations**, learning rate **1e-5**, **16 LoRA layers**, and validation every **10 steps**. The training call was commented out in that notebook, so this is a configuration record rather than a documented successful training run.

To attempt reproduction using the supplied dataset and legacy script:

```bash
python scripts/lora.py \
  --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
  --data data/ \
  --train --iters 100 \
  --steps-per-eval 10 --val-batches -1 \
  --learning-rate 1e-5 --lora-layers 16 \
  --adapter-file adapters_retrained.npz \
  --test
```

The command writes to `adapters_retrained.npz` to **avoid overwriting the supplied checkpoint**. The test option reports token-level loss/perplexity; **it does not automatically compute exact-match command-generation accuracy**. The historical runtime has not been independently reproduced or benchmarked during repository preparation.

## Limitations and next steps

- **No live CLI validation:** The system generates text commands only; their correctness and safety have not been independently verified.
- **No trained classifier:** An earlier exploratory classifier was unfinished and is not included.
- **Limited evaluation:** The saved example outputs are not a quantitative held-out benchmark. Training logs and a verified exact-match score are not available.
- **Narrow data coverage:** The current dataset contains 98 examples within one predefined command grammar.
- **Legacy runtime:** Reproduction requires compatibility with the older Apple MLX scripts; the notebook has not been rerun on Apple Silicon as part of this GitHub cleanup.

Next steps include a stronger held-out command-generation evaluation, expanding the command grammar and datasets, adding argument validation, and testing the approach against a safely isolated network-management environment.

## Credits and licensing notes

The original MLX model, fine-tuning, and generation implementation is adapted from [Apple's MLX examples](https://github.com/ml-explore/mlx-examples); the included files retain Apple's copyright headers. The initial exploratory notebook also drew on a tutorial by [Shaw Talebi](https://youtu.be/3PIqhdRzhxE). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

No blanket license is asserted here over the third-party scripts, pretrained model, or example data. Review their respective license and redistribution requirements before further reuse or public distribution.
