# Supplementary Materials: 

This is the official repository for the supplementary materials for the paper **"“bot lane noob”: Towards Practical Deployment of NLP-based Toxicity Detectors in Video Games"** accepted to **ESORICS 2026 (European Symposium on Research in Computer Security)**.

If you use any of our resources, you are kindly invited to cite our paper:

```bibtex
@inproceedings{ave2026botlane,
    author = {Ave, Jonas and Pekaric, Irdin and Frohner, Matthias and Apruzzese, Giovanni},
    title = {{“bot lane noob”: Towards Practical Deployment of NLP-based Toxicity Detectors in Video Games}},
    booktitle = {European Symposium on Research in Computer Security (ESORICS)},
    year = {2026}
}
```
It contains the following folders and files:

---

## 📂 model_creation

- **L2DTnH Dataset**  
  Contains the dataset used in the paper, consisting of manually annotated League of Legends chat messages labeled as toxic or non-toxic. The dataset was created through a manual annotation process performed by experienced players.

- **Model Training and Evaluation Scripts**  
  Includes the scripts used to fine-tune transformer-based models for toxicity detection using the L2DTnH dataset. The experiments compare our proposed model against several existing toxicity detection models.

- **Experimental Evaluation**  
  Scripts used to reproduce the experimental evaluation presented in the paper, including performance comparisons between our proposed model and other state-of-the-art approaches.

---

## 📂 extension

- **Browser Extension Prototype**  
  Implementation of the browser extension developed in this work. The extension integrates the trained toxicity detection model and analyzes webpage content to detect toxic language related to gaming environments.

- **Local Toxicity Detection**  
  The extension performs all analysis locally in the browser without sending any data to external servers, preserving user privacy while detecting potentially harmful messages.

---

## 📦 Trained Models

The trained models used in our experiments are very large and therefore not included directly in this repository.

To facilitate reproducibility, we provide an extended version of the **model_creation** folder (including the trained models) at the following anonymous link:

https://mega.nz/file/vwkTWSiQ#Kp_SWV0xg2lTAKdf1ELoQFjWwiC91k3bd0MM8rPFI9Q
