# Camouflage Backdoor Attack

This repository contains code for implementing and defending against the Camouflage Backdoor Attack in deep learning models.

## .py .ipynb files:

- `attack_demo.py`: Demonstrates the training process of the backdoor model.
- `camouflage.py`: Provides functions for scaling camouflage.
- `data_loader.py`: Contains the dataset class for poisoned data.
- `trainModel.py`: Provides relevant functions required for training.
- `config.py`: Parameters setting
- `resist-*`: Demonstrate four defenses process and results

## defenses:

The repository includes implementations of the following defense mechanisms:
1. Fine-Pruning
2. Neural Cleanse
3. STRIP
4. Additionally, the related methods of defense CBD is built in `utils.util`.

## Data:

1. **ImageNet: ImageNet Dataset Subset**:
    - Includes a subset of the ImageNet dataset with 10 categories. Category names:
        - n01440764
        - n01496331
        - n01537544
        - n01608432
        - n01632777
        - n01667778
        - n01693334
        - n01729977
        - n01742172
        - n01756291

2. **poison-50-ImageNet-n01537544: Camouflage Images**:
    - 50 images used for camouflage, class: n01537544

3. **cam_maps.csv: Local Features Information**:
    - Information about local features of images used for camouflage.

## saved_models:
- Includes a trained model with ASR (Attack Success Rate) of 96.4% and ACC (Accuracy) of 88.6%.

## logs:
- Defense results of the four defense mechanisms are stored in the `logs` folder.

### Usage:

1. Clone the repository.
2. Run `attack_demo.py` to see the training process of the backdoor model.
3. Use `camouflage.py` to apply scaling camouflage.
4. Use `data_loader.py` for loading the poisoned dataset.
5. Use `trainModel.py` for relevant functions required for training.
