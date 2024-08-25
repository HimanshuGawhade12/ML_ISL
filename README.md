# Indian Sign Language Recognition

Hey there! 👋 Welcome to my Indian Sign Language (ISL) recognition project. This is a fun little machine learning project I put together to recognize hand signs from the ISL alphabet using some Python magic.

## What's Inside?

- **collect_images.py**: This script helps you collect images for each ISL alphabet. It uses your webcam, so just press "Q" when you're ready, and it'll start snapping pics.
- **create_dataset.py**: Once you've got your images, this script processes them and builds a dataset that we'll use for training.
- **model_train.py**: This is where the real action happens. It trains a RandomForestClassifier on the dataset and saves the model.
- **interface.py**: Finally, this script lets you test the model in real-time using your webcam. It'll predict the hand signs you're making!

## How to Use It?

1. **Collect Images**: Run `collect_images.py` and start collecting images for each alphabet.
2. **Create Dataset**: Use `create_dataset.py` to build the dataset.
3. **Train the Model**: Run `model_train.py` to train the model.
4. **Real-time Prediction**: Test it out with `interface.py` and see your hand signs recognized live!

## Dependencies

You'll need Python 3.x and these libraries:
- OpenCV
- MediaPipe
- scikit-learn
- NumPy

