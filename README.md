# Emotion Recognition Using OpenCV and LBPH Face Recognizer

This project is an emotion recognition system implemented using OpenCV and the Local Binary Patterns Histograms (LBPH) face recognizer. It uses a pre-trained Haar Cascade classifier for face detection and the LBPH algorithm for facial emotion classification.

## Features
- Detects faces in images.
- Recognizes facial emotions based on training data.
- Saves and loads training data for future use.

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- OpenCV
- NumPy

Install the dependencies using:
```bash
pip install opencv-python opencv-contrib-python numpy
```

## Folder Structure
```
project-directory/
├── emotion_train/         # Training dataset
├── emotion_val/           # Validation dataset
├── haar_face.xml          # Haar Cascade file for face detection
├── face_trained.yml       # Saved LBPH face recognizer model (generated after training)
├── features.npy           # Saved feature data (generated after training)
├── labels.npy             # Saved label data (generated after training)
├── train.py               # Script to train the model
├── test.py                # Script to test the model
└── README.md              # Project documentation
```

## Training the Model
The training script (`train.py`) processes images in the `emotion_train` directory, extracts facial regions of interest (ROIs), and trains an LBPH face recognizer.

### Steps:
1. Place the training images inside the `emotion_train` directory. Each emotion should have its folder, named after the emotion (e.g., `happy`, `sad`, etc.).
2. Run the `train.py` script:
   ```bash
   python train.py
   ```
3. After training, the model is saved as `face_trained.yml`, and the features and labels are saved as `features.npy` and `labels.npy`, respectively.

### Output:
- **Features shape**: Shape of the features array.
- **Labels shape**: Shape of the labels array.
- **Training time**: Time taken to process the dataset.

## Testing the Model
The testing script (`test.py`) uses the trained model to recognize emotions in new images.

### Steps:
1. Place the test image in the `emotion_val` directory.
2. Update the `img_path` variable in the `test.py` script to point to your test image.
3. Run the `test.py` script:
   ```bash
   python test.py
   ```
4. The script will detect faces, recognize the emotion, and display the image with the predicted emotion and confidence score.

### Output:
- **Label**: Predicted emotion label.
- **Confidence**: Model's confidence score for the prediction.

## Example Output
- Predicted emotion: `sad`
- Confidence: `55.3`

The output image will display the detected face with a rectangle and the predicted emotion.

## Notes
- Ensure the Haar Cascade XML file (`haar_face.xml`) is in the same directory as the scripts.
- The dataset structure should be organized as:
  ```
  emotion_train/
  ├── happy/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── sad/
  │   ├── image1.jpg
  │   └── image2.jpg
  ...
  ```
- Use grayscale images for better performance.

## Troubleshooting
- **Error: File not found**
  Ensure the paths to the dataset, test images, and `haar_face.xml` are correct.
- **Unable to load image**
  Verify that the image format is supported (e.g., `.jpg`, `.png`).

## Acknowledgments
- OpenCV documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Haar Cascade Classifier: Used for face detection.
- LBPH Algorithm: Used for emotion recognition.

Feel free to use, modify, and distribute the code.

