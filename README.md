# DSA 5009 Group Project: Federated Learning on ECG (FedECG)



> Electrocardiogram (ECG) signal classification plays a crucial role in diagnosing and managing various heart diseases. However, sharing labeled data across healthcare institutions for developing accurate classification models is a significant challenge due to privacy concerns and data imbalance issues. To tackle these challenges, we propose a federated learning-based deep learning network for ECG signal classification, which allows multiple healthcare institutions to collaboratively train a shared model without exchanging sensitive patient data.
Our approach aims to develop a robust, accurate, and privacy-preserving ECG classification model that can be employed in real-world healthcare settings. By leveraging federated learning, the model can learn from diverse and unbalanced datasets across various institutions, which is particularly important in the context of heart diseases, as the prevalence and manifestation of these conditions can differ significantly across different populations.
We intend to evaluate our proposed approach using a range of performance metrics, including accuracy, sensitivity, specificity, and F1 score. Additionally, we will compare the performance of our model with state-of-the-art ECG classification models to determine its effectiveness in real-world healthcare settings. Our work contributes to the development of more effective and privacy-preserving ECG analysis, which can ultimately improve patient care and reduce the burden on healthcare providers.
This repository includes:
- Code for the FedECG in our study.


## Environment 
* [PyTorch](https://pytorch.org/) (tested on 1.8.0)



## Datasets
We use the ECG datasets ([link](https://www.kaggle.com/competitions/dsaa5009-spring2023/code)). 

## Usage
To train and evaluate a baseline model, run the following commands:
```
# FedECG
python train_main.py --model RNNAttentionModel --num_workers 5 --device_id 3 

```


## Contact
Please contact [@fan](https://github.com/luckyfan-cs) for questions, comments and reporting bugs.
