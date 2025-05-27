# Design and Implementation of Metabolism Stability Prediction System Based on Graph Neural Networks

**Author**: Yu Zhaolong  
**Affiliation**: East China Jiaotong University

## Abstract

Drug metabolism stability is a critical component in drug development. Traditional prediction methods rely on experimental data and molecular properties but exhibit poor accuracy when handling complex molecular structures. This study designs two GNN-based metabolism stability prediction models that effectively utilize molecular graph structures to enhance prediction accuracy.

Model 1 uses Graph Convolutional Networks (GCN) with multi-layer architecture and global pooling strategies to extract molecular structural features. Model 2 combines Transformer and convolutional techniques with TransformerConv layers and multi-head attention mechanisms to capture long-range molecular interactions and improve generalization capability.

Experimental results show that both models outperform traditional machine learning methods on key metrics such as AUC and AUPR. An interactive web interface has been developed for practical testing and application.

## Model Architectures

### Model 1: GCN-based Architecture
![Model 1 Architecture](images/model1.png)

The GCN model employs:
- Multi-layer Graph Convolutional Network
- Global pooling strategy
- MLP layers for final prediction

### Model 2: Transformer-Enhanced GNN
![Model 2 Architecture](images/model2.png)

The Transformer-GNN model features:
- TransformerConv layers with multi-head attention
- Dynamic attention weight adjustment
- Enhanced long-range interaction modeling

## Interactive Demo

🌐 **Try our models online**: [https://yzl.pythonanywhere.com](https://yzl.pythonanywhere.com)

## Requirements

```
flask==3.1.0
flask-cors==5.0.1
torch==1.13.1
torch-geometric==2.2.0
rdkit==2024.9.6
werkzeug==3.1.3
numpy==1.23.1
pandas==2.2.3
tqdm==4.67.1
scikit-learn==1.1.2
```

## Installation

```bash
git clone https://github.com/yourusername/metabolism-stability-prediction.git
cd metabolism-stability-prediction
pip install -r requirements.txt
```

## Usage

### Run GCN Model
```bash
python main_gcn.py
```

### Run Transformer-GNN Model
```bash
python main_trc.py
```

## Features

- **Two Advanced Models**: GCN-based and Transformer-enhanced GNN architectures
- **High Performance**: Superior results on AUC and AUPR metrics
- **Web Interface**: Interactive platform for easy testing
- **Practical Application**: Ready-to-use for drug development workflows

## Results

Our models demonstrate significant improvements over baseline methods in predicting drug metabolism stability.

## Acknowledgments

Special thanks to **Li Zhongling** for guidance and supervision throughout this project.

## Contact

**Yu Zhaolong**  
East China Jiaotong University  

---

**Interactive Demo**: [https://yzl.pythonanywhere.com](https://yzl.pythonanywhere.com)
