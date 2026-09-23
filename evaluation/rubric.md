# Evaluation rubric

Judge checkpoint contact sheets blind to experiment name. Score each dimension
from 1 (absent/broken) to 5 (strong) and retain the raw scores rather than only
an average.

| Dimension | What to look for |
| --- | --- |
| Style fidelity | Graphite/colored-pencil character, hatching, translucent wash, warm paper, negative space |
| Prompt fidelity | Requested subject, count, action, composition, and palette are present |
| Generalization | Unseen subjects inherit the style without turning into memorized creatures |
| Diversity | Seeds/checkpoints do not collapse to the same framing or anatomy |
| Artifact control | No signatures, UI/screenshot remnants, broken text, border fragments, or severe anatomy failures |
| Trigger locality | Step-0/base and trigger-free outputs remain distinct from triggered LoRA outputs |

Primary selection rule: choose the earliest checkpoint that reaches strong
style fidelity while retaining prompt fidelity and diversity. Training loss is
diagnostic; it is not the selection metric.

Comparison rules:

1. Compare dataset variants within one base-model family first.
2. Use the same prompt and seed at every checkpoint.
3. Treat cross-family Klein/Qwen scores as descriptive because their samplers
   and guidance behavior differ.
4. Keep source-level holdouts fixed. The dataset builder guarantees this when
   `split_seed` is unchanged.
5. Record any manual exclusions before looking at generated results.
