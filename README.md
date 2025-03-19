All batteries included

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

## Quick Start

```bash
poetry install  # Install dependencies
poetry shell    # Activate virtual environment
streamlit run main.py  # Launch app
```

Open http://localhost:8501 in your browser

Archi.

```mermaid
    flowchart TB
        classDef dataNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
        classDef processNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
        classDef storageNode fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
        classDef modelNode fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#6a1b9a
        classDef uiNode fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#c62828
        classDef securityNode fill:#ede7f6,stroke:#4527a0,stroke-width:2px,color:#4527a0

        start([Start]):::processNode --> User[User Interface]:::uiNode
        User --> |"Enter Search Query"| QueryInput[Query Input Box]:::uiNode
        QueryInput --> |"Process Query"| Search[Search Function]:::processNode
        
        subgraph Data["Data Pipeline"]
            direction TB
            RawData[JSON Product Data]:::dataNode --> |"Load"| DataLoad[Load & Preprocess]:::processNode
            DataLoad --> |"Format"| Chunks[Create Text Chunks]:::processNode
        end
        
        Search --> |"Pass Query"| VectorDB[Vector Database]:::storageNode
        Data --> |"Store Data"| VectorDB
        
        subgraph Models["AI Models"]
            direction TB
            SentenceModel[Sentence Transformer]:::modelNode -->|"Create Embeddings"| VectorDB
            FlantT5[FLAN-T5 Generator]:::modelNode -->|"Generate Summary"| Summary[AI Summary]:::processNode
        end
        
        subgraph Security["Security Layer"]
            direction LR
            Encryption[Fernet Encryption]:::securityNode -->|"Encrypt/Decrypt"| VectorDB
        end
        
        VectorDB -->|"Return Matches"| Results[Search Results]:::processNode
        Results -->|"Apply Filters"| EthicalFilter[Ethical Filters]:::processNode
        EthicalFilter -->|"Filtered Results"| Summary
        Summary -->|"Display Results"| Display[UI Display]:::uiNode
        Display -->|"User Views Results"| finish([Finish]):::processNode
        
        style start fill:#81c784,stroke:#388e3c,stroke-width:2px,color:white
        style finish fill:#81c784,stroke:#388e3c,stroke-width:2px,color:white
```