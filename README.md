# Facial Emotion Recognition

Nebras Jafail - GA DSB-3

Predicts the emotion of a face image (angry, disgust, fear, happy, neutral, sad, surprise) using the FER2013 dataset, with a Streamlit app that uses the webcam.

## How to run
1. Download FER2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Unzip it next to the notebook (the notebook reads `archive (5)/train` and `archive (5)/test`).
3. `pip install -r requirements.txt`
4. Run `facial_emotion_recognition.ipynb` (it saves `emotion_cnn.keras`).
5. `streamlit run app.py`

## Results
| Model | Test accuracy |
|---|---|
| Dense (first version) | 36% |
| Dense (fixed) | 34% |
| CNN + EarlyStopping | 41% |

The fixed Dense score is lower because my first score used the test data during training, so it was not fair.

## Mistakes I fixed
- I used the test data as validation, so now I use `validation_split=0.2`
- I loaded gray images as rgb, so now I use `grayscale`
- I trained the same model twice, so now each model is trained once
- I added precision/recall, the loss plot and a CNN

## App
- Upload a photo or use the webcam
- The app finds the face, makes it gray and 48x48, then predicts the emotion

## Limitations
- Weak with bad lighting, side faces or covered faces
- disgust has very few images
- The dataset may not represent everyone equally
