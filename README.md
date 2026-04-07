# Full-Parameter Fine-Tuning for Police Radio ASR
This repository provides a specialized computational pipeline designed for the fine-tuning and evaluation of the OpenAI Whisper-Large-v3 architecture, developed in collaborationg with Dr. Anjalie Field and Kaavya Chaparala @ Johns Hopkins University Center for Language and Speech Processing. 
Our work focuses on the unique acoustic and linguistic challenges found in law enforcement radio communications within the Baltimore and Chicago police districts.

The current implementation builds upon and extends the foundational methodologies established in "Speech Recognition for Analysis of Police Radio Communication" by Tejes Srivastava, Ju-Chieh Chou, Priyank Shroff, Karen Livescu, Christopher Graziul @ University of Chicago.

##
Our approach moves beyond simple fine-tuning by addressing the specific acoustic artifacts of 800MHz trunked radio systems and the linguistic idiosyncraticies of police dispatch. 

Some prominent technical modifications include: 
* Numerical stability: we utilize FP32 training to account for extreme signal spikes, such as sirens and radio interference, present in both audio streams.
* Data architecture: the system is engineered to bridge the gap between legacy speech processing frameworks and modern transformer-based workflows. It is built to ingest Kaldi-style data structures as well as HuggingFace Datasets, providing a flexible pipeline for large-scale ASR research. 
* Evaluation and normalization metrics: given the high frequency of non-standard speech in police communications, our evaluation logic goes beyond basic string matching. We include lexical normalization and reference filtering.
* Memory optimization and numerical stability: to handle the computational demands of the Whisper architecture on high-stakes radio data, the pipeline incorporates specific optimizations for CUDA memory management and gradient calculation. We use expandable segments to more efficiently utilize allocated VRAM on shared GPU nodes, and checkpoint management wherein the system resumes training from the most recent epoch, preventing loss of progress during preemptive scheduling on the cluster.

##
The core of this research utilizes a Seq2Seq objective. to maintain high transcriptive integrity, the loss function is computed by explicitly masking padding tokens. 
This ensures that the model optimization is driven exclusively by the relevant phonetic and lexical content of the police radio transmissions.

The training objective is defined by the following negative log-likelihood:

$$L(\theta) = - \sum_{t=1}^{T} \log P(y_{t} \mid y_{\lt t}, \mathbf{x}; \theta)$$

To balance computational efficiency with the need for high-fidelity transcription, we use the following training configurations:
* Maximum audio duration: 25.0s (audio is truncated beyond this limit to maintain VRAM efficiency).
* Maximum label length: 448 tokens (ensures compatibility with Whisper’s positional embeddings).
* Sampling rate: 16,000 Hz (standardized for high-quality feature extraction via Mel-spectrograms).
* Optimizer: AdamW (implemented via the HuggingFace Seq2SeqTrainer).
* Task type: transcribe (English language-specific fine-tuning).

##
This research handles public safety communications. Users must adhere to ethical guidelines regarding law enforcement data.
