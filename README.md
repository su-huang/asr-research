# Police Radio ASR

This repository provides a specialized computational pipeline designed for the fine-tuning and evaluation of the OpenAI Whisper-Large-v3 and Alibaba Cloud Qwen3-ASR architecture, developed in collaboration with Dr. Anjalie Field and Kaavya Chaparala @ Johns Hopkins University Center for Language and Speech Processing. 

Our work focuses on the unique acoustic and linguistic challenges found in law enforcement radio communications within the Baltimore and Chicago police districts. The current implementation builds upon and extends the foundational methodologies established in *"Speech Recognition for Analysis of Police Radio Communication"* by Tejes Srivastava, Ju-Chieh Chou, Priyank Shroff, Karen Livescu, Christopher Graziul @ University of Chicago.

---

### Pseudolabelling & LLM-as-a-Judge Pipeline
To scale training capabilities beyond scarce human-annotated streams, this pipeline implements an automated pseudolabelling framework coupled with an asynchronous verification layer designed to sanitize downstream training sets. We ran inference with off-the-shelf models across raw audio corpora to generate dense candidate pseudolabels.

While deep autoregressive speech architectures exhibit high resilience when decoding heavily degraded signals, they remain inherently vulnerable to text hallucinations. Within high-noise public safety channels, characterized by persistent 800MHz trunked static, cross-talk, and transient acoustic spikes, these models can generate syntactically pristine but contextually fabricated transcripts.

To prevent structural noise and hallucinations from corrupting the optimization of the fine-tuned Whisper-Large-v3 target model, candidate text sequences pass through an LLM-as-a-Judge verification architecture:

* Linguistic Alignment Evaluation: The LLM judges the semantic validity of the pseudo-labels against specialized law enforcement nomenclature, tight structural constraints (such as regional 10-codes), and cross-turn dispatch context.
* Filtering and Quality Control: Transcripts exhibiting structural mismatches, low confidence distributions, or semantic drift relative to the target domain are programmatically purged before final dataset collation.

This ensures that the loss function during full-parameter, LoRa, and frozen-decoder fine-tuning is driven exclusively by highly authentic, contextually valid lexical features.

---

### Technical Approach & Architecture

Our approach moves beyond simple fine-tuning by addressing the specific acoustic artifacts of 800MHz trunked radio systems and the linguistic idiosyncrasies of police dispatch. 

Key technical modifications and optimizations include: 

* Numerical Stability: We utilize FP32 training to account for extreme signal spikes, such as sirens and radio static interference, present in both audio streams.
* Data Architecture: The system is engineered to bridge the gap between legacy speech processing frameworks and modern transformer-based workflows. It ingests Kaldi-style data structures, maps raw sequence lines to gold-standard alignment manifests, and interfaces cleanly with HuggingFace Datasets to provide a flexible pipeline for large-scale ASR research.
* Evaluation and Normalization Metrics: Given the high frequency of non-standard speech in police communications, our evaluation logic goes beyond basic string matching. We include rigorous lexical normalization and reference filtering.
* Memory Optimization: To handle the computational demands of the Whisper architecture on high-stakes radio data, the pipeline incorporates specific optimizations for CUDA memory management and gradient calculation. We use expandable segments to more efficiently utilize allocated VRAM on shared GPU nodes, alongside robust checkpoint management wherein the system resumes training from the most recent epoch, preventing loss of progress during preemptive scheduling on the cluster.
