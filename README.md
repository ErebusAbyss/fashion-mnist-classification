# Fashion-MNIST Image Classification

## Project Overview
This project is a portfolio piece to demonstrate skills in deep learning (DL) for image classification using TensorFlow and Keras. The goal is to build a convolutional neural network (CNN) to classify clothing items from the Fashion-MNIST dataset. The dataset contains 70,000 grayscale images of 10 clothing categories (e.g., T-shirt, Trouser, Dress).

Key objectives:
- Develop practical experience with DL workflows, including data preprocessing, model building, training, evaluation, and optimization.
- Achieve high accuracy while exploring techniques to improve the model.
- Simulate a real-world ML task for a resume/portfolio.

The best result was achieved with a basic CNN: **90.25% test accuracy**. Attempts to improve it using augmentation, hyperparameter tuning, and transfer learning did not yield significant gains (details below), likely due to the dataset's simplicity (small grayscale images with low variation). This project highlights experimentation and analysis, even when improvements are marginal.

## Dataset
- **Source**: Built-in in TensorFlow (`tf.keras.datasets.fashion_mnist`).
- **Size**: 60,000 training images, 10,000 test images.
- **Format**: Grayscale images (28x28 pixels), 10 classes (labels 0-9).
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

No additional datasets were used.

## Steps and Implementation
The project follows a standard ML pipeline: data loading, EDA, preprocessing, model building, training, evaluation, and optimization attempts.

### 1. Data Loading and EDA
- Loaded the dataset using `tf.keras.datasets.fashion_mnist.load_data()`.
- Shapes: Train (60,000, 28, 28), Test (10,000, 28, 28).
- EDA: Visualized 9 random images and plotted class distribution (balanced, ~6,000 per class).

### 2. Preprocessing
- Normalized pixel values to [0, 1] by dividing by 255.0.
- Reshaped to (samples, 28, 28, 1) for CNN input (added channel dimension for grayscale).

### 3. Model Building
- Used Keras Sequential API for a basic CNN:
  - Conv2D layers with ReLU activation and increasing filters (64, 128, 128).
  - BatchNormalization for stabilization.
  - MaxPooling2D for downsampling.
  - Flatten, Dense (256 with ReLU), Dropout (0.5), output Dense (10 with softmax).
- Compiled with Adam optimizer, sparse_categorical_crossentropy loss, and accuracy metric.

### 4. Training and Evaluation
- Trained with data augmentation (rotation, shift, shear, zoom, flip) using ImageDataGenerator to prevent overfitting.
- Batch size: 128 (later 64 for stability).
- EarlyStopping on val_loss with patience=5.
- Trained for up to 50 epochs; stopped at ~22-25 with test accuracy ~90.25% and loss ~0.2650.
- Visualization: Plotted train/val accuracy to check for overfitting.

### 5. Optimization Attempts
Several techniques were tried to improve accuracy beyond 90.25% (target: 95%+), but gains were minimal (~0-1%), likely because the dataset is simple and the base model was already effective. Details:
- **Data Augmentation**: Added variations to training data (e.g., rotation_range=10-20, flips). Result: Minor improvement to ~90.47%, but no significant boost.
- **BatchNormalization and More Layers**: Added normalization and extra Conv/Dense layers. Result: Stabilized training but accuracy remained ~90%.
- **EarlyStopping**: Prevented overfitting by stopping when val_loss stopped improving. Result: Efficient training, but no accuracy gain.
- **Hyperparameter Tuning with Keras Tuner (Hyperband)**: Automated search for filters (32-256), dense_units (128-512), dropout (0.3-0.5), learning rate (1e-4 to 1e-2). Ran ~30 trials. Best val_accuracy: 90.04%, test: 90.04% (no improvement over base).
- **Transfer Learning with MobileNetV2**: Used pre-trained model on ImageNet, adapted for grayscale (resized to 96x96, repeated channels to RGB). Froze base layers, added custom head. Result: ~58-36% (worse than base due to size mismatch and lack of adaptation).
- **Fine-Tuning Transfer**: Unfroze last layers with low LR (1e-5). Result: ~34-10% (overfitting or weight destruction due to small image size and grayscale).

Conclusions from optimizations: The dataset's low resolution and lack of color limited transfer learning effectiveness. Augmentation and tuning helped stability but not accuracy (base model was near "ceiling" for simple CNN ~92%). For future projects with real images, these techniques would yield bigger gains.

## Results
- Base CNN: Test accuracy 90.25%, loss 0.2650 (best overall).
- Optimized versions: 90.47% max (minor gains from augmentation/tuning).
- Transfer/Fine-tune: 10-58% (not successful due to dataset specifics).

Graphs showed good convergence without severe overfitting.

## How to Run
1. Clone the repo: `git clone https://github.com/ErebusAbyss/fashion-mnist-classification.git`.
2. Create venv: `python -m venv venv; source venv/Scripts/activate`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the notebook: `jupyter notebook fashion_mnist_cnn.ipynb` (or open in VS Code).
5. Models are saved as .keras files (load with `keras.models.load_model('model.keras')`).

Requirements: Python 3.10+, TensorFlow 2.17+, Keras 3.0+ (see requirements.txt).

## Conclusions and Lessons Learned
This project built DL skills: from EDA to advanced optimization. While base accuracy was solid (90.25%), optimizations showed that for simple datasets like Fashion-MNIST, complex techniques (transfer) may not always improve results due to data characteristics (small grayscale images). Key takeaways: Always analyze why methods fail (e.g., size mismatch in transfer) and use cross-validation for robustness. For resume, this demonstrates full ML cycle and experimentation.

Future improvements: Try ResNet50 for transfer or larger datasets like Zalando Fashion.

## Author
- [ErebusAbyss](https://github.com/ErebusAbyss)