# Cross-Lingual Circuit Analysis of Subject-Verb Agreement in Gemma 2B

## Project Overview

This project extends the work of Ferrando & Costa-jussà (2024) on cross-lingual circuit similarity, expanding from 2 languages to 7, adding weight-level analysis, and exploring the connection between mechanistic interpretability and model compression.

**Model**: Google Gemma 2B (18 layers, 8 attention heads per layer, d_model=2048)
**Task**: Subject-verb agreement (SVA) — predicting the correct verb form given a subject's grammatical number

### Languages Analyzed

| Language | Family | Script | Examples | Token Mode |
|----------|--------|--------|----------|------------|
| English | Germanic | Latin | 536 | single-token |
| Spanish | Romance | Latin | 5,684 | single-token |
| French | Romance | Latin | 2,496 | single-token |
| Russian | Slavic | Cyrillic | 260 | single-token |
| Turkish | Turkic | Latin | 18 | first-subword |
| Swahili | Bantu | Latin | 2,366 | first-subword |
| Quechua | Quechuan | Latin | 60 | first-subword |

For languages with agglutinative morphology (Turkish, Swahili, Quechua), plural forms tokenize into multiple subwords in Gemma's tokenizer. We adapted all experiments to use **first-subword token matching**: comparing the first subword token of singular vs plural verb forms, which differ due to prefix changes (Swahili a-/wa-) or stem differences. This extends the methodology beyond the single-token constraint of the original paper.

---

## 1. Activation Patching — Which Components Are Causally Important?

**What this experiment does**: A transformer model processes text through 18 layers, each containing 8 attention heads. We want to know which of these 144 heads are actually responsible for making the model predict the correct verb form. To test this, we use a technique called "activation patching" (also known as denoising).

Here's the idea: we create two versions of the same sentence — one with a singular subject ("The doctor that helped the teacher **is**") and one corrupted with a plural subject ("The doctors that helped the teacher **are**"). We run the model on the corrupted input, which makes it predict the wrong verb. Then, one head at a time, we swap in that head's activation from the clean (correct) run. If swapping in head X's clean activation fixes the prediction, that head is causally important for SVA — it's the one carrying the number signal.

Each cell in the heatmap below shows the "recovery score" for that head: 1.0 = fully restores the correct prediction, 0 = does nothing. This gives us a causal importance map of the entire model.

### English

![Activation Patching — English](results/figures/fig_patching_en.png)

The English patching map shows a clear circuit concentrated in **Layer 13** (L13H0, L13H4, L13H5, L13H7) and **Layer 17** (L17H3, L17H4). L13H7 is the key head identified by Ferrando & Costa-jussà — our replication confirms it. The circuit is sparse: most heads contribute nothing, while a handful carry the entire SVA signal.

### Spanish

![Activation Patching — Spanish](results/figures/fig_patching_es.png)

Spanish shows a strikingly similar pattern to English — the same Layer 13 and Layer 17 heads light up. This confirms the original paper's finding: the SVA circuit is shared across languages.

### French

![Activation Patching — French](results/figures/fig_patching_fr.png)

French (Romance, like Spanish) shows activity in Layers 12-13 and 15, but more diffuse than English. The patching effects are weaker overall — the model is less confident on French SVA, possibly because French verb morphology is less transparent (many forms sound identical when written differently). The circuit uses the same layers but with more heads contributing smaller amounts.

### Russian

![Activation Patching — Russian](results/figures/fig_patching_ru.png)

Russian (Slavic, Cyrillic script) activates Layers 13-15 with moderate patching effects. Despite the completely different script and language family, the circuit occupies the same region of the network as English and Spanish. This is evidence that the SVA circuit location is determined by the model architecture, not by language-specific features.

### Turkish

![Activation Patching — Turkish](results/figures/fig_patching_tr.png)

Turkish (Turkic, SOV word order, agglutinative) shows a broader activation pattern across many heads. With only 18 examples (the smallest dataset), individual patching effects are noisier. The circuit appears more distributed — Turkish SVA involves more heads with smaller individual contributions, possibly because the model has less specialized circuitry for this low-resource agglutinative language.

### Swahili

![Activation Patching — Swahili](results/figures/fig_patching_sw.png)

Swahili (Bantu, prefix-based agreement) shows moderate patching effects concentrated in Layers 13-15. Despite being a completely different language family with prefix-based (not suffix-based) agreement, the circuit occupies the same network region. This suggests the model uses a universal "agreement" circuit regardless of whether agreement is encoded through prefixes or suffixes.

### Quechua

![Activation Patching — Quechua](results/figures/fig_patching_qu.png)

Quechua (Quechuan, SOV, agglutinative) with only 60 examples shows detectable but weak patching effects. The model likely has minimal Quechua training data, yet the SVA signal still routes through Layers 12-14. Even for a language the model barely knows, it attempts to use the same circuit architecture.

### Cross-Language Comparison

Across all 7 languages from 5 language families, the patching maps share a common structure: **Layer 13 consistently contains the highest-importance heads**. The circuit is more concentrated (fewer heads, higher values) for well-resourced languages (EN, ES) and more distributed (more heads, lower values) for low-resource languages (TR, QU). This suggests the model develops specialized, efficient circuits for languages it knows well, while using broader, weaker circuits for unfamiliar languages.

---

## 2. Direct Logit Attribution — Directional Contributions

**What this experiment does**: The model's final prediction is a sum of contributions from every component (each attention head and each MLP layer). Direct Logit Attribution (DLA) mathematically decomposes this sum to show exactly how much each component pushes the prediction toward the correct verb vs the incorrect verb.

Think of it like this: if the model outputs "is" with confidence 5.0 and "are" with confidence 2.0, the difference is 3.0. DLA breaks that 3.0 into individual contributions: head A contributed +2.1, head B contributed +1.5, head C contributed -0.6, etc. Components with large positive values are helping the model get the right answer; components with negative values are working against it.

The bar charts below show the top 15 heads by absolute contribution for each language. Blue bars push toward the correct verb, red bars push away from it.

### English and Spanish

![DLA — English](results/figures/fig_dla_en.png)
![DLA — Spanish](results/figures/fig_dla_es.png)

In both languages, L13H7 and L17H4 have the largest positive DLA (pushing toward the correct verb). Some heads show negative DLA — they actually push toward the *wrong* verb, acting as opposing forces in the circuit.

### French

![DLA — French](results/figures/fig_dla_fr.png)

French DLA shows a more balanced distribution — more heads contribute moderate amounts rather than a few heads dominating. L13H7 still appears but is not as dominant as in English, suggesting French SVA relies on a more distributed computation.

### Russian

![DLA — Russian](results/figures/fig_dla_ru.png)

Russian DLA shows strong contributions from L13H7 and L17H4, similar to English, but with additional contributions from L15 heads. The Cyrillic-script language uses the same key heads as Latin-script languages, confirming these are language-independent processing units.

### Turkish

![DLA — Turkish](results/figures/fig_dla_tr.png)

Turkish DLA contributions are spread across many heads with no single dominant head, consistent with the distributed patching pattern. The model doesn't have a specialized Turkish SVA circuit — it uses a general-purpose set of heads.

### Swahili

![DLA — Swahili](results/figures/fig_dla_sw.png)

Swahili shows moderate DLA values with L13H7 and L17H4 still prominent. The prefix-based agreement system (a- for singular, wa- for plural) produces a detectable signal through the same heads that handle English suffix-based agreement.

### Quechua

![DLA — Quechua](results/figures/fig_dla_qu.png)

Quechua DLA shows weak but detectable contributions, consistent with the model's limited training data for this language.

---

## 3. Weight Importance Maps — Which Weights Are Wired for SVA?

**What this experiment does**: The previous experiments (patching, DLA) look at what happens during inference — they require running data through the model. This experiment is fundamentally different: it analyzes the model's **weight matrices directly**, without running any data.

Each attention head has two key weight matrices: W_V (what information to read from the input) and W_O (what to write to the output). Together, the OV circuit (W_V × W_O) defines what the head does. We decompose this matrix and ask: "how aligned is this head's OV circuit with the verb-number direction?" — i.e., the direction in the model's internal space that separates "is" from "are" in the output vocabulary.

The result is a per-head importance score that reflects the model's **structural wiring** for the SVA task. A high score means the head's weights are physically oriented to process number agreement, regardless of whether it's actually activated during inference. This is the basis for our per-weight importance maps (150M individually-scored weights per language).

The heatmaps below show this weight-level importance across all layers and heads.

### English

![Weight Importance — English](results/figures/fig_weight_importance_en.png)

The English weight map shows L0H5 (importance 5.77) and L0H3 (3.31) as the most weight-aligned heads, followed by L17H4 (1.47), L15H1 (1.43), and L13H7 (1.02). The Layer 0 dominance is notable — these embedding-layer heads have weights that are structurally oriented toward the verb-number direction, even though their causal importance (from patching) is lower.

### Spanish

![Weight Importance — Spanish](results/figures/fig_weight_importance_es.png)

Spanish shows the same L0H5/L0H3 dominance but with much higher values (19.15 and 15.96). The scale difference from English suggests the unembedding direction for Spanish verb pairs is more aligned with these early-layer weight matrices.

### French

![Weight Importance — French](results/figures/fig_weight_importance_fr.png)

French weight importance (max 20.13 at L0H5) closely mirrors Spanish — both Romance languages have nearly identical weight importance profiles. This makes sense: their verb systems are related, so the unembedding directions for French and Spanish verb pairs point in similar directions in the model's internal space.

### Russian

![Weight Importance — Russian](results/figures/fig_weight_importance_ru.png)

Russian (max 10.41 at L0H3) shows lower values than the Romance languages but the same Layer 0 dominance. Notably, L0H3 edges out L0H5 in Russian — the only language where this happens. The Cyrillic verb tokens may have unembedding vectors that align slightly differently with the Layer 0 heads.

### Turkish

![Weight Importance — Turkish](results/figures/fig_weight_importance_tr.png)

Turkish (max 17.81 at L0H3) shows high values despite being a low-resource language. This is a weight-level analysis — it doesn't depend on how much Turkish data the model saw, only on how the weight matrices relate to the Turkish verb unembedding direction. The weights are structurally aligned even if the model rarely activates them for Turkish.

### Swahili

![Weight Importance — Swahili](results/figures/fig_weight_importance_sw.png)

Swahili has the second-highest values of any language (max 39.39 at L0H5). This may seem paradoxical for a low-resource language, but it reflects the structure of Swahili verb tokens in the embedding space — the a-/wa- prefix distinction produces unembedding vectors that are strongly aligned with Layer 0 head weights.

### Quechua

![Weight Importance — Quechua](results/figures/fig_weight_importance_qu.png)

Quechua shows the highest values overall (max 39.56 at L0H5), similar to Swahili. Both are agglutinative languages with distinctive morphological markers that produce well-separated unembedding vectors.

### Cross-Language Comparison

**Key finding**: L0H5 and L0H3 are the top 2 heads in every single language. The weight importance maps are more uniform across languages than the patching maps — the model's weight structure contains a universal SVA-aligned subspace that all languages share, even though the activation-level circuits vary.

The scale varies dramatically: English peaks at 5.77 while Quechua peaks at 39.56. This doesn't mean Quechua has a "stronger" circuit — it means the Quechua verb token pairs have unembedding vectors that are more aligned with the Layer 0 weight matrices. The relative pattern (which heads are important) is consistent; the absolute scale depends on the language's tokenization.

---

## 4. SVD Spectra — Functional Rank of Each Head

**What this experiment does**: Each attention head's OV matrix can be decomposed into independent "channels" using Singular Value Decomposition (SVD). Think of SVD as breaking the head's computation into ranked components: the first singular value captures the most important direction, the second captures the next most important orthogonal direction, and so on.

If a head has one very large singular value and the rest are tiny, it's essentially doing one thing — a simple, clean computation (like "check if the subject is singular or plural"). If the singular values are all similar, the head is doing many things simultaneously, making it harder to interpret and harder to compress.

For model compression, heads with peaked spectra (one dominant direction) are easier to approximate with low-rank matrices. Heads with flat spectra need to keep more of their weight matrix intact.

![SVD Spectrum — English](results/figures/fig_svd_spectrum_en.png)

L13H7's spectrum is relatively flat — the top singular value explains only 11.6% of the total. This means L13H7 is doing more than just a simple number-direction projection; it's implementing a multi-dimensional computation. This complicates the "one direction per head" narrative from the original paper.

---

## 5. Edge Attribution Patching — Fast Circuit Discovery

**What this experiment does**: Activation patching (Section 1) is thorough but slow — it requires a separate forward pass for every head we want to test (144 heads × number of examples). Edge Attribution Patching (EAP) achieves nearly the same result using calculus: instead of physically swapping activations, it computes the gradient of the model's output with respect to each component's activation. This tells us, mathematically, how sensitive the output is to changes in each component.

The key advantage: EAP needs only 2 forward passes + 1 backward pass per example (compared to 144+ forward passes for full patching). This makes it 10-100x faster and lets us include MLP layers in the analysis, not just attention heads.

The comparison below shows EAP scores for all components (both attention heads and MLPs) side by side across all 7 languages.

![EAP Comparison](results/figures/fig_eap_comparison.png)

The EAP comparison across all 7 languages reveals that **MLP layers dominate**: MLP17 is the #1 component in every language, followed by MLP16 and MLP13. This is invisible in the patching analysis (which only patches attention heads) and highlights that the MLP layers are doing critical work for SVA.

---

## 6. Attention Patterns — What Does Each Head Attend To?

**What this experiment does**: Attention heads work by "looking at" certain positions in the input and combining information from them. For SVA, the key question is: when the model is about to predict the verb at the end of the sentence, which heads look back at the subject noun?

We extract the attention weights from the final position (where the verb is predicted) for every head, then measure how much attention falls on the subject noun positions (positions 1-3 in the sentence). A head that puts most of its attention on the subject is likely reading the number signal (singular vs plural).

The heatmaps below show "subject attention score" for each head — darker cells mean the head attends more to the subject.

![Subject Attention — English](results/figures/fig_attention_en.png)
![Subject Attention — Spanish](results/figures/fig_attention_es.png)
![Subject Attention — French](results/figures/fig_attention_fr.png)
![Subject Attention — Russian](results/figures/fig_attention_ru.png)
![Subject Attention — Turkish](results/figures/fig_attention_tr.png)
![Subject Attention — Swahili](results/figures/fig_attention_sw.png)
![Subject Attention — Quechua](results/figures/fig_attention_qu.png)

**Key finding**: **L13H7** is the #1 subject-attending head in English (0.465 attention to subject positions), confirming it reads the subject number signal. But **L1H7** is the most consistent subject-reader across languages — it's #1 in ES, TR, SW, and QU. This suggests a two-stage circuit: L1H7 reads the subject early, L13H7 processes the number signal at the critical layer.

---

## 7. Logit Lens — Where Does the Prediction Form?

**What this experiment does**: A transformer processes information layer by layer, building up an internal representation (called the "residual stream"). At the very end, this representation is converted into word probabilities by the unembedding matrix. The logit lens applies this conversion at every intermediate layer — like taking snapshots of what the model "thinks" the next word is at each step of processing.

This reveals where the correct verb prediction forms. If the correct verb appears at Layer 5, the earlier layers have already solved the problem. If it only appears at Layer 17, the model needs nearly all its layers. For model compression, layers where the prediction hasn't changed are candidates for removal.

The plots below show two metrics at each layer:
- **Blue line (logit diff)**: How much the model's internal state favors the correct verb over the incorrect one. Higher = more confident.
- **Red line (P(correct))**: The actual probability of the correct verb if we decoded at that layer. This stays near zero until the very end because the model's vocabulary is 256K tokens — even a large logit diff translates to a tiny probability until the final LayerNorm concentrates it.

![Logit Lens — English](results/figures/fig_logit_lens_en.png)
![Logit Lens — Spanish](results/figures/fig_logit_lens_es.png)
![Logit Lens — French](results/figures/fig_logit_lens_fr.png)
![Logit Lens — Russian](results/figures/fig_logit_lens_ru.png)
![Logit Lens — Turkish](results/figures/fig_logit_lens_tr.png)
![Logit Lens — Swahili](results/figures/fig_logit_lens_sw.png)
![Logit Lens — Quechua](results/figures/fig_logit_lens_qu.png)

The blue line (logit diff) shows the model's internal confidence in the correct verb grows through the layers, with a large jump at Layer 13-14 and peaking at Layer 17. But the red line (P(correct)) stays near zero until the very last layer — the model's logit diff is large but spread across a huge vocabulary. **LayerNorm at the output performs critical rescaling** that converts a broad signal into an actual peaked prediction.

---

## 8. Cross-Lingual Steering — Does the English Direction Transfer?

**What this experiment does**: The PCA analysis (from the original paper) found that attention head L13H7 encodes subject number as a single direction in its output space — one end means "singular", the other means "plural". We extract this direction from English data only (the PC1 vector).

The steering experiment then asks: if we take this English-derived "number direction" and inject it into the model while it processes a Spanish/French/Russian/etc. sentence, does it flip the verb prediction? If yes, the model uses the same internal representation for number across languages.

We vary the steering magnitude α from 0 (no intervention) to 50 (strong intervention). The flip rate measures what fraction of sentences change their verb prediction. A flip rate increasing with α is evidence of causal cross-lingual transfer.

![Steering — Spanish](results/figures/fig_steering_es.png)
![Steering — French](results/figures/fig_steering_fr.png)
![Steering — Russian](results/figures/fig_steering_ru.png)
![Steering — Turkish](results/figures/fig_steering_tr.png)
![Steering — Swahili](results/figures/fig_steering_sw.png)
![Steering — Quechua](results/figures/fig_steering_qu.png)

The flip rate increases with steering magnitude α in all languages, confirming cross-lingual causal transfer. Russian shows the strongest effect (27-30% flip rate at α=50) despite using Cyrillic script and being from a different language family.

---

## 9. Circuit Knockout — Is the Circuit Necessary and Sufficient?

**What this experiment does**: The previous experiments identified a set of "important" heads. But are they actually the complete circuit? Two tests:

1. **Necessity test**: Zero out all the identified circuit heads simultaneously. If the model can no longer do SVA, the circuit was necessary — you can't remove it without breaking the task.

2. **Sufficiency test**: Zero out every head *except* the identified circuit. If the model can still do SVA, the circuit is sufficient — it contains everything needed, and everything else is irrelevant for this task.

A circuit that passes both tests is validated as a complete, minimal description of how the model solves SVA. This is important for compression: if the circuit passes sufficiency, you know exactly which heads to keep and which to prune.

![Circuit Knockout Summary](results/figures/fig_knockout_summary.png)

- **Blue bars**: Baseline accuracy (no ablation)
- **Red bars**: Accuracy after ablating the circuit (testing necessity — lower is better, means the circuit was needed)
- **Green bars**: Accuracy after ablating everything except the circuit (testing sufficiency — higher is better, means the circuit alone is enough)

**Turkish** achieves the cleanest validation: ablating the 28-head circuit drops accuracy from 72% to 44% (necessity PASS), and keeping only those heads preserves 83% accuracy (sufficiency PASS). English and Spanish show WEAK results because the patching threshold (0.1) identified only 2 heads — the real circuit is larger.

---

## 10. CKA Convergence — Where Do Languages Converge?

**What this experiment does**: When the model processes "The doctor is" (English) and "El doctor es" (Spanish), the internal representations at each layer are different — they start from different tokens. But at some point, the model must converge to a shared representation of "singular subject, needs singular verb". CKA (Centered Kernel Alignment) measures how similar the representations are between language pairs at each layer.

CKA is invariant to rotation and scaling — it measures whether the *geometry* of the representations (the relative distances between examples) is similar, not whether the raw vectors are identical. A CKA of 1.0 means the representations have the same structure; 0 means no structural similarity.

![CKA Convergence](results/figures/fig_convergence.png)

Cross-lingual similarity peaks at **Layer 13** (mean CKA = 0.26), the exact layer containing the key SVA head. Representations converge at the critical layer then diverge again. This is evidence that the model develops a shared, language-independent representation for grammatical number at the layer where it needs it.

---

## 11. Cross-Layer Connection Map

**What this experiment does**: In a transformer, heads don't communicate directly — they read from and write to a shared "residual stream". Head A in Layer 5 writes something to the residual stream, and Head B in Layer 13 reads from it. The connection strength between A and B depends on how much A's output overlaps with what B reads.

We compute this for every pair of heads: "if Head A writes its task-relevant output to the residual stream, how much does Head B pick up from that to produce its own task-relevant output?" This gives us a wiring diagram of the circuit — not based on activations, but on the actual weight matrices.

The result is a connection matrix showing which heads feed information to which other heads for the SVA task.

**English** routes through **L13→L17**: L13H3→L17H4 (0.41), L13H7→L17H4 (0.37)

**All other languages** route through **L14→L16**: L14H0→L16H1 and L14H3→L16H0 dominate

English is the outlier in circuit wiring — the same task, the same model, but different information routing. This may be because English dominates the training data, leading the model to develop a specialized processing path, while all other languages share a common default route.

---

## 12. Circuit Map vs Wanda — Do Different Methods Agree?

**What this experiment does**: There are two philosophies for deciding which weights matter in a neural network:

1. **Our approach (circuit map)**: Analyze the weight matrices mathematically and ask "which weights are structurally wired to contribute to this specific task?" This uses the OV decomposition and task projection from Section 3.

2. **Wanda (standard baseline)**: Run data through the model and observe which weights get activated with large values. Importance = |weight| × ||activation||. This is the standard approach used in model pruning.

If both methods agree on which weights are important, then our expensive circuit analysis doesn't add much over the simpler Wanda baseline. If they disagree, our method captures something Wanda misses — structural task relevance that isn't visible from activations alone.

We compared the two at multiple thresholds across all 7 languages.

| Metric | Value |
|--------|-------|
| Spearman rank correlation | ρ = 0.74 (moderate) |
| Top 10% Jaccard overlap | 0.30-0.35 |
| **Disagreement rate at top 10%** | **68.1%** |
| Circuit-only critical weights | 27.3M |
| Wanda-only critical weights | 27.3M |
| Both agree critical | 25.5M |

**The two methods identify substantially different sets of critical weights.** At the top 1% (most critical), overlap drops to just 19-35%. Our circuit-derived importance captures task-relevant weight structure that activation-based methods miss, and vice versa. This suggests that combining both signals could produce better compression decisions than either alone.

Per-weight maps are available for all 7 languages, covering 150,994,944 attention weights each.

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
