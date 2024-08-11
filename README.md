# CoffeeBeanQualityDetection
Project for detecting coffee bean quality using YOLOv8. Includes model training, deployment on NVIDIA Jetson Nano, and conversion to TensorRT for optimized inference.

## Table of Contents
- [Setup](#setup)
- [Installation](#installation)
- [Training](#training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Setup

Ensure that you have Conda installed on your system. This project uses a Conda virtual environment for package management.

```bash
conda create --name yolov8 python=3.8
conda activate yolov8
```

## Installation

1. (Install CUDA and cuDNN according to your GPU specifications.)[https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html#]
2. (Install PyTorch, ensuring they are compatible with the CUDA version you have installed.)[https://pytorch.org/]
3. Install the Ultralytics YOLOv8 package
```bash
pip install ultralytics
```

## Training

Use the `train_yolov8_model.py` script to train your models. This script handles the entire training pipeline, including data loading, model initialization, and saving the trained model.

```bash
python train_yolov8_model.py --data-path /path/to/your/data --epochs 50 --batch-size 16
```

## Deployment

After training, you can deploy the model on an NVIDIA Jetson Nano. Convert the trained model to TensorRT for optimized inference.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to adjust any part of this to suit your specific project details.
