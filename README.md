# Cascading and Proxy Membership Inference Attack (NDSS 2026)

This repository contains the official implementation of two novel membership inference attacks (MIAs) presented in our NDSS 2026 paper:

- **CMIA (Cascading Membership Inference Attack)**: An adaptive attack that incorporates membership dependencies through iterative cascading rounds
- **PMIA (Proxy Membership Inference Attack)**: A non-adaptive attack that uses proxy data for likelihood ratio testing

## Overview

### CMIA: Cascading Membership Inference Attack

CMIA is a new **attack-agnostic** framework that leverages membership dependencies for attacks. The attack operates through multiple cascading iterations, where each round uses memberships from previous rounds to improve attack performance.

**Key Features:**
- **Iterative Cascading**: Multiple rounds of attack with membership information propagation
- **Anchor-based Training**: Uses high-confidence predictions as anchors for conditional shadow model training
- **Adaptive Thresholding**: Dynamically determines positive and negative thresholds using reference models
- **Membership Dependencies**: Incorporates relationships between sample membership statuses

### PMIA: Proxy Membership Inference Attack

PMIA is a **non-adaptive** attack that uses proxy data to perform likelihood ratio testing without requiring access to the target model's training data distribution.

**Key Features:**
- **Proxy-based Approach**: Uses adversary's own data as proxy for likelihood ratio computation
- **Multiple Proxy Variants**: 
  - **Global Proxy**: Uses all adversary's data as a single proxy
  - **Class-based Proxy**: Uses same-class instances as proxy
  - **Instance-based Proxy**: Uses similar instances as proxy
- **Non-adaptive**: Does not require access to the target model's training data for shadow training

## Code Architecture

### CMIA Framework

```
CMIA/
├── run_shadow.py          # Shadow model training
├── run_anchor.py          # Anchor generation and threshold determination
├── run_attack.py          # Main attack execution
├── shadow.py              # Shadow model implementation
├── anchor.py              # Anchor-based training logic
├── attacks/               # Attack implementations
├── config/                 # Configuration files for different datasets
├── models/                # Neural network architectures
└── utils/                 # Utility functions
```

### PMIA Framework

```
PMIA/
├── run_shadow.py          # Shadow model training
├── run_attack.py          # Main attack execution
├── shadow.py              # Shadow model implementation
├── attack.py              # Attack orchestration
├── attacks/               # Attack implementations
├── config/                 # Configuration files
├── models/                # Neural network architectures
└── utils/                 # Utility functions
```

## Usage

### CMIA (Cascading Membership Inference Attack)

CMIA operates through multiple cascading rounds. Each round consists of three main steps:

#### 1. Shadow Model Training

```bash
cd CMIA

# Train shadow models for the current round
python run_shadow.py --dataset cifar10 --cur_round 1
```

#### 2. Anchor Generation and Threshold Determination

```bash
# Generate anchors and determine thresholds for the next round
python run_anchor.py --dataset cifar10 --cur_round 1 --attack lira
```

#### 3. Attack Execution

```bash
# Run the attack for the current round
python run_attack.py --dataset cifar10 --cur_round 1 --attack lira

# Continue for additional rounds...
```


### PMIA (Proxy Membership Inference Attack)

PMIA requires shadow model training followed by attack execution with different proxy variants:

#### 1. Shadow Model Training

```bash
cd PMIA

# Train shadow models
python run_shadow.py --dataset cifar10
```

#### 2. Attack Execution

Choose one of the three proxy variants:

```bash
# Global proxy (using all adversary's data)
python run_attack.py --dataset cifar10 --attack pglobal

# Class-based proxy (using same class instances)
python run_attack.py --dataset cifar10 --attack pclass

# Instance-based proxy (using similar instances)
python run_attack.py --dataset cifar10 --attack pinstance
```

## Configuration

### Dataset Configuration

Both frameworks support multiple datasets through configuration files:

- `cifar10.yml`: CIFAR-10 dataset configuration
- `cifar100.yml`: CIFAR-100 dataset configuration
- `mnist.yml`: MNIST dataset configuration
- `fmnist.yml`: Fashion-MNIST dataset configuration


## Citation

Please cite our paper if our work is useful for your research:

```bibtex
@inproceedings{du2026cascading,
  title={Cascading and Proxy Membership Inference Attack},
  author={Du, Yuntao and Li, Jiacheng and Chen, Yuetian and Zhang, Kaiyuan and Yuan, Zhizhen and Xiao, Hanshen and Ribeiro, Bruno and Li, Ninghui},
  booktitle={Proceedings of the Network and Distributed System Security Symposium (NDSS)},
  year={2026}
}
```
