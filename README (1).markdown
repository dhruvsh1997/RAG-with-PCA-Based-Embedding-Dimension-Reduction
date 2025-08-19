# RAG with PCA-Based Embedding Dimension Reduction

![Banner](https://via.placeholder.com/1200x200?text=RAG+with+PCA+Embedding+Reduction)

## Overview
This repository hosts a Jupyter Notebook exploring a Retrieval-Augmented Generation (RAG) pipeline that tests the impact of reducing embedding dimensions using Principal Component Analysis (PCA) before storing vectors in a FAISS vector database. The experiment evaluates whether aggressive dimension reduction (384D to 18D) maintains effective retrieval and generation quality, potentially improving computational efficiency.

**Repository Name**: `rag-pca-embedding-reduction`

**Key Features**:
- **Embedding Model**: Sentence Transformers (`all-MiniLM-L6-v2`) for 384D embeddings.
- **Dimension Reduction**: PCA to compress embeddings to 18D.
- **Vector Store**: FAISS with a custom PCA embedding wrapper.
- **Reranking**: MS-MARCO Cross-Encoder for refined retrieval.
- **LLM**: Groq-powered Llama-3.3-70B for answer generation.
- **Test Document**: `solid-python.pdf` (SOLID principles in Python).
- **Goal**: Assess the viability of reduced embeddings in RAG pipelines.

Ideal for researchers and developers optimizing RAG systems for resource-constrained environments.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Flowchart](#flowchart)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)

## Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/rag-pca-embedding-reduction.git
cd rag-pca-embedding-reduction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**:
```
langchain
sentence-transformers
faiss-cpu
pypdf
groq
langchain-community
langchain-groq
scikit-learn
```

**Prerequisites**:
- **Groq API Key**: Set via environment variable (`GROQ_API_KEY`) or Google Colab secrets.
- **Test PDF**: Place `solid-python.pdf` in the root directory or update the notebook path.

Run the notebook:
```bash
jupyter notebook PCABasedDimentionReducedEmbedding-RAG.ipynb
```

## Usage
1. **Load Document**: Import and split `solid-python.pdf` into chunks (500 characters, 50 overlap).
2. **Generate Embeddings**: Create 384D embeddings with Sentence Transformers.
3. **Reduce Dimensions**: Apply PCA to compress embeddings to 18D.
4. **Build Vector Store**: Use FAISS with a custom PCA embedding wrapper.
5. **Query**: Ask questions (e.g., "What is the main objective of the document?").
6. **Retrieve & Rerank**: Fetch top-k chunks and refine with Cross-Encoder.
7. **Generate Answers**: Use the LLM to produce answers and compare pre/post-reranking results.

Example query:
```python
question = "What is the main objective of the document?"
```

## Pipeline Architecture
The pipeline consists of:
1. **Document Processing**: Load and split PDF into chunks.
2. **Embedding Generation**: Create 384D vectors using Sentence Transformers.
3. **PCA Reduction**: Compress embeddings to 18D.
4. **Vector Store**: Store reduced embeddings in FAISS.
5. **Retrieval**: Fetch relevant chunks using FAISS retriever.
6. **Reranking**: Reorder chunks with Cross-Encoder for relevance.
7. **Generation**: Format context with `ChatPromptTemplate` and query the LLM.
8. **Evaluation**: Compare answers before and after reranking.

This tests whether reduced embeddings maintain semantic accuracy for retrieval.

## Flowchart
```mermaid
flowchart TD
    A[Load PDF] --> B[Split into Chunks]
    B --> C[Generate 384D Embeddings<br>(Sentence Transformers)]
    C --> D[PCA Reduction<br>(384D → 18D)]
    D --> E[Store in FAISS<br>(Custom PCA Wrapper)]
    E --> F[User Query]
    F --> G[Retrieve Top-K Chunks]
    G --> H[Rerank with Cross-Encoder]
    H --> I[Format Prompt<br>(ChatPromptTemplate)]
    I --> J[Generate Answer<br>(Groq LLM)]
    J --> K[Compare Pre/Post-Reranking]
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px
```

## Experiment Details
- **PCA Choice**: PCA captures maximum variance in fewer dimensions, reducing from 384D to 18D (based on sample count).
- **Custom Embeddings**: `PCAEmbeddings` class ensures PCA is applied consistently for indexing and querying.
- **Reranking**: Cross-Encoder scores query-chunk pairs to improve retrieval relevance.
- **Test Setup**:
  - **Document**: ~22 chunks from `solid-python.pdf`.
  - **Question**: Tests high-level document understanding.
  - **Evaluation**: Qualitative comparison of answers.
- **Objective**: Validate if reduced embeddings lower storage/compute costs without sacrificing RAG performance.

Code Snippet (PCA Embedding Wrapper):
```python
class PCAEmbeddings(Embeddings):
    def __init__(self, model, pca):
        self.model = model
        self.pca = pca

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return self.pca.transform(vectors).tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode([text], convert_to_numpy=True)
        return self.pca.transform(vector)[0].tolist()
```

## Results
- **Pre-Reranking**: Reduced embeddings retrieve relevant chunks, but ranking may be suboptimal.
- **Post-Reranking**: Cross-Encoder improves chunk relevance, leading to more accurate LLM answers.
- **Observation**: PCA reduction to 18D retains sufficient semantic information for RAG, with reranking compensating for retrieval noise.

## Limitations and Future Work
- **Limitations**:
  - PCA assumes linear relationships; non-linear methods (e.g., UMAP) may improve results.
  - Small dataset; larger corpora needed for robust testing.
- **Future Work**:
  - Add quantitative metrics (e.g., ROUGE, BLEU).
  - Experiment with other reduction techniques (t-SNE, Autoencoders).
  - Test on diverse documents and questions.
  - Optimize PCA components using explained variance.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Submit a pull request with clear descriptions.

Report issues or suggest enhancements via the Issues tab.

## License
MIT License. See [LICENSE](LICENSE) for details.