# full-parameter fine-tuning for police radio asr
this repository provides a specialized computational pipeline designed for the fine-tuning and evaluation of the **OpenAI Whisper-Large-v3** architecture, developed in collaborationg with **Dr. Anjalie Field** and **Kaavya Chaparala** @ **Johns Hopkins University Center for Language and Speech Processing**. our work focuses on the unique acoustic and linguistic challenges found in law enforcement radio communications within the Baltimore and Chicago police districts.

the current implementation builds upon and extends the foundational methodologies established in the paper:
> *"Speech Recognition for Analysis of Police Radio Communication"* by **Tejes Srivastava, Ju-Chieh Chou, Priyank Shroff, Karen Livescu, Christopher Graziul** @ **University of Chicago**.

##
our approach moves beyond simple fine-tuning by addressing the specific acoustic artifacts of 800MHz trunked radio systems and the linguistic idiosyncraticies of police dispatch. 

some prominent technical modifications include: 
* **numerical stability**: we utilize **Full-Precision (FP32) training** to account for extreme signal spikes, such as sirens and radio interference, present in both audio streams.
* **data architecture**: the system is engineered to bridge the gap between legacy speech processing frameworks and modern transformer-based workflows. it is built to ingest **Kaldi-style** data structures as well as **HuggingFace Datasets**, providing a flexible pipeline for large-scale ASR research. 
* **evaluation and normalization metrics**: given the high frequency of non-standard speech in police communications, our evaluation logic goes beyond basic string matching. we include **lexical normalization** and **reference filtering**.
* **memory optimization and numerical stability**: to handle the computational demands of the Whisper architecture on high-stakes radio data, the pipeline incorporates specific optimizations for CUDA memory management and gradient calculation. we use **expandable segments** to more efficiently utilize allocated VRAM on shared GPU nodes, and **checkpoint management** wherein the system resumes training from the most recent epoch, preventing loss of progress during preemptive scheduling on the cluster.

##
the core of this research utilizes a **Seq2Seq** (Sequence-to-Sequence) objective. to maintain high transcriptive integrity, the loss function is computed by explicitly masking padding tokens ($ID = -100$). this ensures that the model optimization is driven exclusively by the relevant phonetic and lexical content of the police radio transmissions.

the training objective is defined by the following negative log-likelihood:

$$L(\theta) = - \sum_{t=1}^{T} \log P(y_{t} \mid y_{\lt t}, \mathbf{x}; \theta)$$

to balance computational efficiency with the need for high-fidelity transcription, we use the following training configurations:
* **maximum audio duration:** 25.0s (audio is truncated beyond this limit to maintain VRAM efficiency).
* **maximum label length:** 448 tokens (ensures compatibility with Whisper’s positional embeddings).
* **sampling rate:** 16,000 Hz (standardized for high-quality feature extraction via Mel-spectrograms).
* **optimizer:** AdamW (implemented via the HuggingFace `Seq2SeqTrainer`).
* **task type:** transcribe (English language-specific fine-tuning).

##
this research handles public safety communications. this model includes experimental normalization to help mitigate the transcription of PII. users must adhere to ethical guidelines regarding law enforcement data.
