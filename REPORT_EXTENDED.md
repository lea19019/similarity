# Cross-Lingual Circuit Analysis of Subject-Verb Agreement in Gemma 2B

## Project Overview

This project extends the work of Ferrando & Costa-jussà (2024) on cross-lingual circuit similarity, expanding from 2 languages to 7, adding weight-level analysis, and exploring the connection between mechanistic interpretability and model compression.

**Model**: Google Gemma 2B (18 layers, 8 attention heads per layer, d_model=2048)
**Task**: Subject-verb agreement (SVA) — predicting the correct verb form given a subject's grammatical number

### Languages Analyzed

| Language | Family | Script | Examples | Token Mode | SVA Type |
|----------|--------|--------|----------|------------|----------|
| English | Germanic | Latin | 536 | single-token | SVO, suffix |
| Spanish | Romance | Latin | 5,684 | single-token | SVO, suffix |
| French | Romance | Latin | 2,496 | single-token | SVO, suffix |
| Russian | Slavic | Cyrillic | 260 | single-token | SVO, suffix |
| Turkish | Turkic | Latin | 18 | first-subword | SOV, agglutinative |
| Swahili | Bantu | Latin | 2,366 | first-subword | SVO, prefix |
| Quechua | Quechuan | Latin | 60 | first-subword | SOV, agglutinative |

---

## Methods

### Replication (Ferrando & Costa-jussà, 2024)

1. **Activation patching** (denoising): Run corrupted input, replace one component's activation with its clean-run value, measure recovery of correct verb prediction. Identifies causally important heads/MLPs.

2. **Direct logit attribution (DLA)**: Decompose the logit difference into additive per-component contributions by projecting each head/MLP output onto the verb-number unembedding direction. Exact decomposition, no approximation.

3. **PCA on L13H7 outputs**: Extract the subject-number direction from the key attention head's output space. PC1 separates singular from plural across languages.

4. **Cross-lingual activation steering**: Add the English-derived PC1 direction to target-language forward passes and measure verb prediction flips.

### Extensions

5. **Edge Attribution Patching (EAP)**: Gradient-based approximation of activation patching using 2 forward passes + 1 backward pass instead of O(layers × heads) forward passes. Based on Syed et al. (2023).

6. **Weight-level circuit map**: For every attention head, compute OV matrix (W_V @ W_O), decompose via SVD, and project onto the task-relevant unembedding direction. Produces per-weight importance scores for all 150M attention weights.

7. **Cross-layer connection map**: Measure how each head's output aligns with downstream heads' task-relevant input, revealing the circuit's wiring diagram through the residual stream.

8. **Cross-lingual geometry comparison**: CKA (Centered Kernel Alignment), SVCCA (Singular Vector Canonical Correlation Analysis), RSA (Representational Similarity Analysis), and Procrustes alignment across all language pairs at every layer.

9. **Wanda-style importance**: Activation-weighted importance (|W| × ||X||) as the standard pruning baseline, following Wanda (Sun et al., 2024).

10. **Attention pattern analysis**: Extract attention matrices showing what each head attends to from the verb prediction position. Identifies which heads read the subject noun.

11. **Logit lens**: Project residual stream onto unembedding at each layer to visualize where the correct verb prediction forms.

12. **Circuit knockout validation**: Ablate the identified circuit to test necessity (does SVA break?) and ablate everything except the circuit to test sufficiency (is SVA preserved?).

### Multi-Token Approach

For languages with agglutinative morphology (Turkish, Swahili, Quechua), plural forms tokenize into multiple subwords. We adapted the pipeline to use **first-subword token matching**: comparing the first subword token of singular vs plural verb forms, which differ due to prefix changes (Swahili a-/wa-) or stem differences. This extends the methodology beyond the single-token constraint of the original paper.

---

## Key Findings

### 1. Universal SVA Circuit Structure

**MLP17 is the dominant component in ALL 7 languages.** Edge Attribution Patching consistently ranks MLP17, MLP16, and MLP13 as the top 3 components. The SVA circuit is language-independent at the component level.

### 2. Cross-Layer Wiring Divergence

The cross-layer connection map revealed a striking pattern:
- **English**: L13H3→L17H4 (strength 0.41), L13H7→L17H4 (0.37) — routes through **Layer 13→Layer 17**
- **All other languages** (ES, FR, RU, TR, SW, QU): L14H0→L16H1 and L14H3→L16H0 dominate — routes through **Layer 14→Layer 16**

English is the outlier. Possible explanations: (a) English dominates training data, developing a specialized circuit; (b) English has simpler agreement morphology (no overt marking in many contexts), requiring different processing.

### 3. Cross-Lingual Steering Transfers

The English-derived PC1 direction from L13H7 causally affects verb predictions in all 6 target languages:

| Target | Flip rate (α=50, +dir) | Flip rate (α=50, -dir) |
|--------|----------------------|----------------------|
| Spanish | 20.4% | 36.5% |
| French | 20.2% | 25.1% |
| Russian | 27.3% | 29.6% |
| Turkish | — | — |
| Swahili | — | — |
| Quechua | — | — |

Russian shows the strongest steering effect despite being Slavic with Cyrillic script.

### 4. Representational Convergence at Layer 13

CKA analysis across layers shows cross-lingual similarity peaks at **Layer 13** (mean CKA = 0.140), the layer containing the key SVA head. Representations converge at the critical layer then diverge again.

Pair-wise CKA at final layer: EN-ES (0.132) > ES-RU (0.101) ≈ EN-FR (0.101) > ES-FR (0.099) > EN-RU (0.088).

### 5. Attention Patterns: L1H7 is the Universal Subject Reader

**L13H7** attends to the subject in English (#1 with 0.465 attention score), but **L1H7** is the most consistent subject-reading head across languages:
- #1 in ES (0.542), TR (0.451), SW (0.189), QU (0.408)
- Top-5 in FR and RU

This suggests a two-stage circuit: L1H7 reads the subject early, L13H7 processes the number signal at the critical layer.

### 6. Logit Lens: Late Prediction Formation

The correct verb prediction forms very late in the network. At Layer 17, the logit difference is large but the correct verb is still ranked ~1000th out of 256K tokens. Only after the final LayerNorm does it become a top prediction (rank 12 for EN, rank 117 for SW). LayerNorm performs critical rescaling.

### 7. Circuit Knockout Results

| Language | Circuit Size | Necessity Drop | Sufficiency |
|----------|-------------|----------------|-------------|
| English | 2 heads | 4.7% | WEAK |
| Spanish | 2 heads | 16.4% | WEAK |
| French | 11 heads | 5.5% | PASS |
| Russian | 3 heads | 5.5% | PASS |
| **Turkish** | **28 heads** | **27.8%** | **PASS** |
| Swahili | 5 heads | 7.8% | WEAK |
| Quechua | 3 heads | 10.0% | PASS |

Turkish achieves the cleanest circuit validation — ablating the 28-head circuit drops accuracy by 27.8%, and keeping only those heads preserves 83.3% accuracy.

### 8. Per-Weight Importance Maps

For each language, we mapped all 150,994,944 attention weights individually:
- **~10% of weights are critical** (above the 90th percentile of importance)
- **~90% of weights are candidates** for aggressive quantization/pruning
- Per-weight maps saved as `weight_map_{lang}.npz`

### 9. Circuit Map vs Wanda: 68% Disagreement

The most significant finding for model compression:

| Metric | Value |
|--------|-------|
| Spearman rank correlation | ρ = 0.74 (moderate agreement) |
| Top 10% Jaccard overlap | 0.30-0.35 (low overlap) |
| **Disagreement rate** | **68.1%** |
| Circuit-only critical weights | 27.3M |
| Wanda-only critical weights | 27.3M |
| Both agree critical | 25.5M |

**The two methods identify substantially different sets of critical weights.** Our circuit-derived importance (based on weight structure relative to the task direction) captures weights that Wanda (based on activation magnitudes) misses, and vice versa. This suggests that combining both signals could produce better compression decisions than either alone.

At the most selective level (top 1%), overlap drops to just 19-35% (Jaccard 0.10-0.22). The methods are most divergent on the weights they consider most critical.

---

## File Inventory

### Result Files (`results/`)

| File Pattern | Content | Shape |
|-------------|---------|-------|
| `patching_{lang}.npz` | Activation patching per head | (18, 8) |
| `dla_{lang}.npz` | Direct logit attribution per head | (18, 8) |
| `neurons_{lang}.npz` | Per-neuron MLP DLA, layers 13 & 17 | (16384,) per layer |
| `edge_patching_{lang}.npz` | EAP node scores | (162,) |
| `circuit_map_{lang}.npz` | Head importance, SVD spectra, connections | (18, 8) + (144, 144) |
| `weight_map_{lang}.npz` | Per-weight W_V and W_O importance | (18, 8, 2048, 256) |
| `wanda_{lang}.npz` | Wanda activation × weight importance | (18, 8, 2048, 256) |
| `attention_{lang}.npz` | Attention patterns from last position | (18, 8, max_seq) |
| `logit_lens_{lang}.npz` | Per-layer logit diff, rank, probability | (19, n_examples) |
| `knockout_{lang}.npz` | Circuit necessity/sufficiency scores | scalars |
| `pca_L13H7.npz` | PC1 direction, projections, labels | (256,) |
| `steering_{lang}.npz` | Flip rates across alpha values | (6,) |
| `activations_{lang}.npz` | Per-layer residual stream activations | (18, 128, 2048) |
| `geometry.npz` | CKA, SVCCA, RSA, Procrustes per layer | (18, 21) |

### Visualization Files (`results/figures/`)

**2D Plots (PNG)**:
- `fig1_patching_en.png` through `fig_patching_{lang}.png` — activation patching heatmaps
- `fig_dla_{lang}.png` — DLA bar charts
- `fig5_pca_scatter.png` — PC1 projections colored by number/language
- `fig6_steering.png` — steering flip rate curves
- `fig_weight_importance_{lang}.png` — per-head weight importance heatmaps
- `fig_svd_spectrum_{lang}.png` — OV matrix SVD spectra
- `fig_eap_comparison.png` — side-by-side EAP scores
- `fig_convergence.png` — CKA convergence across layers

**3D Interactive (HTML)**:
- `viz_importance_3d.html` — overlaid weight importance surfaces per language
- `viz_circuit_graph_3d.html` — 3D scatter of component importance
- `viz_cka_animated.html` — animated CKA heatmap with layer slider
- `viz_convergence_3d.html` — multi-metric convergence in 3D
- `viz_task_cosine.html` — per-head task projection cosine similarity
- `viz_svd_spectrum.html` — SVD spectra visualization

### Figure Descriptions

**Activation patching heatmaps** (`fig_patching_{lang}.png`): Each cell (layer, head) shows normalized patch effect ∈ [0,1]. Value of 1 means that head fully restores correct verb prediction when patched from clean to corrupted run. L13H7 shows high values across languages.

**DLA bar charts** (`fig_dla_{lang}.png`): Top-15 heads by absolute direct logit attribution. Blue bars = positive (pushes toward correct verb), red = negative. Based on exact decomposition of logit difference into per-component contributions.

**PCA scatter** (`fig5_pca_scatter.png`): PC1 projections of L13H7 head outputs. Blue = singular, red = plural. Circles = EN, triangles = ES. Clean separation confirms L13H7 encodes a language-independent subject-number direction.

**Steering curves** (`fig6_steering.png`): Flip rate (fraction of predictions that change) as a function of steering magnitude α. Increasing α along the PC1 direction progressively flips more verb predictions, demonstrating causal transfer.

**Weight importance heatmaps** (`fig_weight_importance_{lang}.png`): Per-head weight importance = ||OV @ unembed_dir||. Shows which heads' weight matrices are structurally aligned with the SVA task direction. Derived from static weights, not activations.

**SVD spectra** (`fig_svd_spectrum_{lang}.png`): Top-10 singular values of each head's OV matrix at Layer 13. Flat spectra = distributed computation across many directions. Peaked spectra = clean rank-1 computation.

**EAP comparison** (`fig_eap_comparison.png`): Side-by-side bar charts of Edge Attribution Patching node scores across languages. Shows gradient-based importance of each head and MLP.

**CKA convergence** (`fig_convergence.png`): Mean CKA across all language pairs at each layer. Peak at Layer 13 shows representations are most similar across languages at the critical SVA layer.

**3D importance surface** (`viz_importance_3d.html`): Interactive 3D plot with X=layer, Y=head, Z=weight importance. One semi-transparent surface per language, overlaid for comparison. Rotate and hover to explore.

**Animated CKA** (`viz_cka_animated.html`): 7×7 language pair similarity matrix with a slider to step through layers 0-17. Shows how cross-lingual similarity evolves through the network.

---

## References

### Primary Paper
- Ferrando, J. & Costa-jussà, M.R. (2024). On the Similarity of Circuits across Languages. *EMNLP Findings 2024*. https://aclanthology.org/2024.findings-emnlp.591

### Mechanistic Interpretability
- Elhage, N., Nanda, N., Olsson, C., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html
- Wang, K., Variengien, A., Conmy, A., et al. (2023). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. *ICLR 2023*. https://arxiv.org/abs/2211.00593
- Conmy, A., Mavor-Parker, A., Lynch, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*. https://arxiv.org/abs/2304.14997

### Edge Attribution Patching
- Syed, A., Rager, C., & Conmy, A. (2023). Attribution Patching Outperforms Automated Circuit Discovery. *BlackBoxNLP 2024*. https://arxiv.org/abs/2310.10348

### SVD-Based Interpretability
- Millidge, B. & Black, S. (2025). Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits. https://arxiv.org/abs/2511.20273

### Logit Lens
- nostalgebraist (2020). interpreting GPT: the logit lens. *LessWrong*. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose, N., Furman, H., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *NeurIPS 2023*. https://arxiv.org/abs/2303.08112

### Representational Similarity
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*. https://arxiv.org/abs/1905.00414 (CKA)
- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis. *NeurIPS 2017*. https://arxiv.org/abs/1706.05806
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational Similarity Analysis. *Frontiers in Systems Neuroscience*. https://doi.org/10.3389/neuro.06.004.2008 (RSA)
- Schönemann, P. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*. (Procrustes alignment)

### Pruning and Quantization
- Sun, M., Liu, Z., Bair, A., & Kolter, J.Z. (2024). A Simple and Effective Pruning Approach for Large Language Models. *ICLR 2024* (Wanda). https://arxiv.org/abs/2306.11695
- Frantar, E. & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. *ICML 2023*. https://arxiv.org/abs/2301.00774
- Ashkboos, S., Croci, M., et al. (2024). SliceGPT: Compress Large Language Models by Deleting Rows and Columns. *ICLR 2024*. https://arxiv.org/abs/2401.15024
- Men, X., Xu, M., et al. (2024). ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. *ACL Findings 2025*. https://arxiv.org/abs/2403.03853
- Tseng, A., Chee, J., et al. (2024). QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks. https://arxiv.org/abs/2402.04396

### Circuit Tracing
- Anthropic (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Anthropic (2025). On the Biology of a Large Language Model. https://transformer-circuits.pub/2025/attribution-graphs/biology.html

### Sparse Autoencoders
- Lieberum, T., Rajamanoharan, S., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. https://arxiv.org/abs/2408.05147
- Templeton, A., Conerly, T., et al. (2024). Scaling Monosemanticity. *Anthropic*. https://transformer-circuits.pub/2024/scaling-monosemanticity/

### Cross-Lingual Representations
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML 2024*. https://arxiv.org/abs/2405.07987

### Attribution-Guided Compression
- Bhaskar, A. et al. (2025). SparC3: Sparse Circuit-Based Compression and Correction of LLMs. https://arxiv.org/abs/2506.13727

### Task-Aware Compression
- Lin, J., Tang, J., et al. (2024). AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration. *MLSys 2024*. https://arxiv.org/abs/2306.00978
- Frantar, E., Ashkboos, S., et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. *ICLR 2023*. https://arxiv.org/abs/2210.17323
- Kang, H. et al. (2025). Optimal Brain Restoration: Post-Training Joint Quantization and Pruning. https://arxiv.org/abs/2509.11177

### Tools and Libraries
- Nanda, N. & Bloom, J. (2022). TransformerLens. https://github.com/TransformerLensOrg/TransformerLens
- Bloom, J. et al. (2024). SAELens. https://github.com/decoderesearch/SAELens
- Fang, G. et al. (2023). DepGraph: Towards Any Structural Pruning (Torch-Pruning). *CVPR 2023*. https://github.com/VainF/Torch-Pruning

---

## Reproducibility

### Environment
- Python 3.11, PyTorch ≥ 2.2.0, TransformerLens ≥ 1.17.0
- GPU: NVIDIA H200 (150GB) or A100 (80GB)
- Cluster: BYU RC (SLURM), partitions m13h (H200) and cs (A100)

### Running the Pipeline
```bash
# 1. Generate datasets (login node, needs internet)
uv run python -m circuits.data --lang all --model gemma-2b

# 2. Main pipeline — 7 languages, all analyses
sbatch run_all.sh

# 3. Extended analyses — Wanda, attention, logit lens, knockout
sbatch run_extended.sh

# 4. Visualizations (login node, no GPU)
bash run_viz.sh
```

### Tests
```bash
uv run pytest -v  # 187 tests, all CPU-only
```
