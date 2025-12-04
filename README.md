# MNIST-dataset-CNN

This repo trains a simple Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset using PyTorch.

The project is intentionally split into small, clean files so you can clearly see the role of **data**, **model**, **training**, and **utilities**.

---

## Project Structure

```bash
.
├── data.py      # Data loading: transforms + MNIST DataLoaders
├── model.py     # CNN model definition (Net)
├── utils.py     # Helper functions: device, checkpoint, accuracy
├── train.py     # train() and test() functions
├── main.py      # Entry point: wires everything together and runs training
└── README.md
```
---

## Requirements

* Python 3.8+
* PyTorch
* Torchvision

---

## Active Extension: 

The pipeline functions as a classifier, but it is presently configured to train the model and report overall metrics instead of showing individual image predictions.
