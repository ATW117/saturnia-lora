# Saturnia LoRA

**A reproducible style study for FLUX.2 Klein Base 4B.** Saturnia explores how different mixes of reference and synthetic images affect a botanical pencil-and-wash illustration style. This repository contains the experiment compiler, evaluation prompts, local inference UI, and a small visual preview. **Training images, synthetic datasets, checkpoints, and LoRA adapters are not published here.**

## A small preview

| FLUX.2 Klein Base 4B | + Saturnia LoRA · step 2,000 |
| :--: | :--: |
| <img src="docs/preview/moth-beetle-base.jpg" alt="Step-zero base model moth-beetle" width="420"> | <img src="docs/preview/moth-beetle-lora.jpg" alt="Moth-beetle with the all-synthetic Saturnia LoRA" width="420"> |

**Moth-beetle.** The base image is the original step-zero sample `1788610328293__000000000_0.jpg`; the right image uses the `05_all_synthetic` adapter at step 2,000. Both use the prompt “A small moth-beetle creature with folded wings and rootlike feet, centered against an open background” with `SATURNIA_STYLE`, seed `314159`, 28 steps, and guidance `4.0`.

| Example | Base model | + Saturnia LoRA |
| :-- | :--: | :--: |
| **Vivid bird-frog** · seed `314159` | <img src="docs/preview/vivid-frog-base.jpg" alt="Base model vivid bird-frog" width="270"> | <img src="docs/preview/vivid-frog-lora.jpg" alt="Vivid bird-frog with the synthetic-core Saturnia LoRA" width="270"> |
| **Walking bromeliad** · seed `314159` | <img src="docs/preview/walking-bromeliad-base.jpg" alt="Base model walking bromeliad" width="270"> | <img src="docs/preview/walking-bromeliad-lora.jpg" alt="Walking bromeliad with Saturnia LoRA" width="270"> |
| **Caterpillar with bluebells** · seed `314159` | <img src="docs/preview/caterpillar-bluebells-base.jpg" alt="Base model caterpillar carrying bluebells" width="270"> | <img src="docs/preview/caterpillar-bluebells-lora.jpg" alt="Caterpillar carrying bluebells with Saturnia LoRA" width="270"> |
| **Botanical dolphin-spider** · seed `42` | <img src="docs/preview/dolphin-spider-base.jpg" alt="Base model botanical dolphin creature with spider legs" width="270"> | <img src="docs/preview/dolphin-spider-lora.jpg" alt="Botanical dolphin creature with spider legs and Saturnia LoRA" width="270"> |
| **Railway station** · seed `314159` | <img src="docs/preview/railway-station-base.jpg" alt="Base model railway station" width="270"> | <img src="docs/preview/railway-station-lora.jpg" alt="Railway station in Saturnia pencil style" width="270"> |
| **Beetle ornate A** · seed `314159` | <img src="docs/preview/beetle-ornate-a-base.jpg" alt="Base model ornate beetle with botanical shell and sprout" width="270"> | <img src="docs/preview/beetle-ornate-a-lora.jpg" alt="Ornate beetle with botanical shell and sprout generated with Saturnia LoRA" width="270"> |
| **Snail patterned B** · seed `314159` | <img src="docs/preview/snail-patterned-b-base.jpg" alt="Base model many-eyed patterned snail" width="270"> | <img src="docs/preview/snail-patterned-b-lora.jpg" alt="Many-eyed patterned snail generated with Saturnia LoRA" width="270"> |
| **Azure snail-frog** · seed `314159` | <img src="docs/preview/azure-snail-frog-base.jpg" alt="Base model azure snail-frog hybrid" width="270"> | <img src="docs/preview/azure-snail-frog-lora.jpg" alt="Azure snail-frog hybrid generated with Saturnia LoRA" width="270"> |
| **Orange fungal tortoise** · seed `314159` | <img src="docs/preview/orange-fungal-tortoise-base.jpg" alt="Base model orange tortoise carrying a mushroom garden" width="270"> | <img src="docs/preview/orange-fungal-tortoise-lora.jpg" alt="Orange tortoise carrying a mushroom garden generated with Saturnia LoRA" width="270"> |
| **Violet cicada-human** · seed `314159` | <img src="docs/preview/violet-cicada-human-base.jpg" alt="Base model violet cicada-human with cyan wings" width="270"> | <img src="docs/preview/violet-cicada-human-lora.jpg" alt="Violet cicada-human with cyan wings generated with Saturnia LoRA" width="270"> |
| **Golden spider weaver** · seed `314159` | <img src="docs/preview/golden-spider-weaver-base.jpg" alt="Base model golden spider-human weaving turquoise threads" width="270"> | <img src="docs/preview/golden-spider-weaver-lora.jpg" alt="Golden spider-human weaving turquoise threads generated with Saturnia LoRA" width="270"> |

The vivid frog uses the same prompt on both sides: `SATURNIA_STYLE. A long-legged bird-frog in vivid cobalt, magenta, turquoise, and saffron.` Both images use seed `314159`, 28 steps, and guidance `4.0`; the base is the step-zero sample and the right image uses `02_synthetic_core` at step 2,000. The bromeliad, caterpillar, railway station, and six additional examples use `06_everything` at step 2,000 and LoRA strength `1.0`.

The dolphin-spider pair uses the prompt `SATURNIA_STYLE. A small botanical dolphin creature on warm paper with spider legs.`, seed `42`, 28 steps, and guidance `4.0` on both sides. The right image uses `06_everything` at strength `0.8`. These are selected visual examples, not a quantitative benchmark.

## What is in the repository

- `src/saturnia_lora/`: source validation, deterministic per-source holdouts, caption preparation, AI Toolkit config compilation, and the CLI.
- `configs/`: six dataset compositions, a fixed training protocol, and model profiles. FLUX.2 Klein 4B is the current focus; the Qwen profile is retained for a later comparison.
- `evaluation/`: fixed prompts and a qualitative checkpoint rubric.
- `saturnia_ui.py` and `compiled_lora.py`: a local Gradio UI and optional compiled LoRA inference path for the Klein setup.
- `docs/preview/`: only the twenty-four generated images displayed above.

The six experiments span reference only, synthetic only, and combined datasets. The local study had 161 accepted image/caption pairs before holdouts. Dataset files are intentionally excluded, so a fresh clone can inspect the catalog and tests but cannot prepare or train until matching local sources are supplied.

## Use the experiment compiler

Requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
make test
saturnia-lora list
```

Place your own paired `.png`/`.jpg` and `.txt` caption files under the source paths named in the selected TOML experiment. For example, `01_reference_only` reads `images/`. Each caption should begin with `SATURNIA_STYLE,` followed by a content sentence; with the default `content_only` policy, later explicit style sentences are removed from the prepared caption. The original datasets are private; use data you have rights to train on.

To prepare and compile a Klein run:

```bash
saturnia-lora prepare 01_reference_only
saturnia-lora compile --experiment 01_reference_only --model flux2_klein_4b
```

Prepared datasets, manifests, configs, samples, and checkpoints are written under `.runs/`, which Git ignores. The model profile defaults to `black-forest-labs/FLUX.2-klein-base-4B`; set `FLUX2_KLEIN_BASE_PATH` to a local Diffusers-format checkpoint if needed. The **Base** 4B checkpoint is the fine-tuning target. See the [official FLUX.2 repository](https://github.com/black-forest-labs/flux2) and [Klein training guide](https://docs.bfl.ai/flux_2/flux2_klein_training).

Training requires a separately installed [AI Toolkit](https://github.com/ostris/ai-toolkit) and its GPU dependencies. `scripts/bootstrap_ai_toolkit.sh /path/to/ai-toolkit` clones its source; follow that checkout's installation instructions for your machine. Then run:

```bash
saturnia-lora run \
  --experiment 01_reference_only \
  --model flux2_klein_4b \
  --ai-toolkit-dir /path/to/ai-toolkit
```

To print the Klein commands for all six experiments, use `saturnia-lora matrix --model flux2_klein_4b --ai-toolkit-dir /path/to/ai-toolkit`. Run metadata records the trainer revision and resolved paths. The training protocol holds rank, alpha, learning rate, steps, sampling seed, and checkpoint cadence fixed across dataset variants; see [the architecture notes](ARCHITECTURE.md) for the design and [the rubric](evaluation/rubric.md) for checkpoint selection.

## Local inference UI

The UI expects a compatible local AI Toolkit installation, its Python environment with Gradio and Pillow, the Klein Base 4B checkpoint, a CUDA GPU, and adapter files under `.runs/output/<run-name>/`. It binds to `127.0.0.1` and disables model downloads by default:

```bash
AI_TOOLKIT_DIR=/path/to/ai-toolkit bash scripts/run_saturnia_ui.sh
```

The UI can also generate from the base model with no adapter. Set `SATURNIA_COMPILE_LORA=0` to disable the optional compiled LoRA path if your PyTorch/compiler setup does not support it. No adapters are provided by this repository.
