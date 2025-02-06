# KG-LLaVA: Reproducibility for AAAI Paper

## Overview
This repository provides the official implementation of **"LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies" (AAAI 2025).** KG-LLaVA integrates Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) with Vision-Language Models (VLMs) to generate Natural Language Explanations (NLEs) for medical imaging.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
  - [MIMIC-NLE](#mimic-nle)
  - [RadGraph Triplet Extraction](#radgraph-triplet-extraction)
  - [Datastore Retrieval](#datastore-retrieval)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citing](#citing)

## Installation
To set up the environment, run:
```bash
git clone https://github.com/yourusername/AAAI-Reproducibility.git
cd AAAI-Reproducibility
pip install -r requirements.txt
```

To set up the MedCLIP environment separately:
```bash
pip install git+https://github.com/RyanWangZf/MedCLIP.git
pip install faiss-gpu
```

## Dataset

### MIMIC-NLE
To obtain the **MIMIC-NLE dataset**, follow the instructions at:
- [MIMIC-NLE Repository](https://github.com/maximek3/MIMIC-NLE)
- Download MIMIC-CXR reports: [PhysioNet](https://physionet.org/content/mimic-cxr/2.1.0/)

We use the training dataset following the **official MIMIC-NLE split**.

### RadGraph Triplet Extraction
To generate **Knowledge Graph (KG) triplets** from medical reports, we utilize **RadGraph**. Follow the instructions below:

1. Download and set up RadGraph from: [Stanford-AIMI RadGraph](https://github.com/Stanford-AIMI/radgraph)
2. Run the inference script to extract triplets:
```bash
python dataset_preparation.py
```
3. We filter triplets to retain only those with **suggestive_of** relationships.

### Datastore Retrieval
A datastore is built using **MedCLIP** and FAISS to facilitate knowledge retrieval. The datastore includes:
- `kg_nle_index`
- `kg_nle_index_captions.json`

To retrieve triplets for test images, use:
```bash
python datastore_retrieval.py
```

## Training
To prepare the dataset for LLaVA training, execute the following:
```bash
python dataset_preparation.py
```
This ensures the dataset is in the required format:
```json
[
    {
        "id": "0",
        "split": "train",
        "image": "p11/p11941242/s50000014/dffc8ab2-ff37704f-2fb29e6d-51e08075-88bca914.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nThe image-specific triplets from the knowledge graph are: opacities suggestive_of effusions; TRIPLETS HERE BASED ON KG;. And for the given image, Which signs show that the patient has uncertain Atelectasis, positive Lung Opacity, uncertain Pleural Effusion, uncertain Pneumonia?"
            },
            {
                "from": "gpt",
                "value": "Retrocardiac opacity with silhouetting of the left hemidiaphragm and lateral border of the descending aorta is nonspecific and could reflect any of a combination of atelectasis, focal pneumonia or even a small effusion."
            }
        ]
    }
]
```

To train **LLaVA** with KG triplets, run:
```bash
bash KG-LLaVA/models/LLaVA/scripts/v1_5/finetune_task_lora.sh
```

## Evaluation
For model evaluation, use:
```bash
bash KG-LLaVA/models/LLaVA/scripts/v1_5/eval/vqav2.sh
```


## Acknowledgements
This repository makes use of the following open-source projects:
- **[LLaVA](https://github.com/haotian-liu/LLaVA)**
- **[MedCLIP](https://github.com/RyanWangZf/MedCLIP)**
- **[RadGraph](https://github.com/Stanford-AIMI/radgraph)**
- **[MIMIC-NLE](https://github.com/maximek3/MIMIC-NLE)**
- **[SmallCap](https://github.com/RitaRamo/smallcap)**

## Citation
This work will appear in AAAI-2025 soon.
If you use this work, please cite:
```bibtex
@article{hamza2024llava,
  title={LLaVA Needs More Knowledge: Retrieval Augmented Natural Language Generation with Knowledge Graph for Explaining Thoracic Pathologies},
  author={Hamza, Ameer and Ahn, Yong Hyun and Lee, Sungyoung and Kim, Seong Tae and others},
  journal={arXiv preprint arXiv:2410.04749},
  year={2024}
}
```
