# FlashEKGR

**FlashEKGR: Fast Training for Embedding-Based Knowledge Graph Reasoning**

FlashEKGR is a high-performance training framework for **Embedding-based Knowledge Graph Reasoning (EKGR)** models.
It introduces multi-level system optimizations to significantly accelerate EKGR training without sacrificing accuracy.

Key techniques include:

* **Logical Operator Parallelism (LOP)** – eliminates redundant computation across query branches
* **Staleness-Free Pipeline (SFP)** – overlaps embedding loading with model computation without introducing stale embeddings
* **Layout-Aware Kernel (LAK)** – improves GPU memory access efficiency
* **Dynamic CUDA Graph Caching (DGC)** – reduces kernel launch overhead for dynamic workloads

Extensive experiments demonstrate **2.06× – 5.67× speedup** over the state-of-the-art framework SMORE.

---

# Table of Contents

* Overview
* Installation
* Dataset Preparation
* Citation

---

# Overview

Embedding-based Knowledge Graph Reasoning (EKGR) models embed entities, relations, and logical queries into a unified vector space for reasoning.

However, training EKGR models suffers from several system bottlenecks:

* redundant logical operator computation
* CPU–GPU data transfer latency
* inefficient GPU memory access
* high kernel launch overhead

FlashEKGR addresses these issues through **algorithm–system co-design**.

Supported EKGR models include:

* BetaE
* Query2Box
* GQE
* ComplEx
* DistMult

Datasets used in the paper include:

* FB15k
* NELL995
* FB400k
* Freebase86m
* Wiki90m

---

# Installation

## Requirements

The project has been tested with the following environment:

| Component | Version               |
| --------- | --------------------- |
| Python    | ≥ 3.9                 |
| CUDA      | 11.7                  |
| PyTorch   | 2.0.1                 |
| GCC       | ≥ 9                   |
| GPU       | NVIDIA A40 or similar |

---

## 1. Clone the Repository

```bash
git clone https://github.com/BUPT-Reasoning-Lab/FlashEKGR.git
cd FlashEKGR
```

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
conda create -n flashekgr python=3.9
conda activate flashekgr
```

---

## 3. Install LibTorch

FlashEKGR requires the LibTorch distribution compiled for CUDA 11.7.

Download and extract:

```bash
wget https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.1%2Bcu117.zip
unzip libtorch-shared-with-deps-2.0.1%2Bcu117.zip
```

Set the environment variable:

```bash
export LIBTORCH_PATH=/path/to/libtorch
```

Example:

```bash
export LIBTORCH_PATH=$HOME/libtorch
```

---

## 4. Initialize Submodules

```bash
git submodule update --init --recursive
```

---

## 5. Build and Install

```bash
pip install -e .
```

---

## 6. Verify Installation

```bash
python -c "import flashekgr"
```

If no error occurs, the installation is successful.

---

# Dataset Preparation

Download datasets used in the paper.

Example (FB15k):

```bash
bash scripts/download_fb15k.sh
```

Dataset directory structure:

```
datasets/
   FB15k/
   NELL995/
   FB400k/
   Freebase86m/
   Wiki90m/
```

---

# Citation

If you find this project useful, please cite our paper:

```
@inproceedings{flashekgr,
  title={FlashEKGR: Fast Embedding-Based Knowledge Graph Reasoning Models Training},
  author={Wentai Zhang, Teng Xu, Weiguang Wang, Junxing Li, Jun Zhang, Yifan Zhu and Haihong E},
  booktitle={ICDE},
  year={2026}
}
```
