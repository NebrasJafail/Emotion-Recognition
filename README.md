# Facial Emotion Recognition

Nebras Jafail - GA DSB-3

Predicts the emotion of a face image (angry, disgust, fear, happy, neutral, sad, surprise) using the FER2013 dataset, with a Streamlit app that uses the webcam.

## How to run
1. Download FER2013 from Kaggle and put the `train` and `test` folders next to the notebook.
2. `pip install -r requirements.txt`
3. Run `facial_emotion_recognition.ipynb` (it saves `emotion_cnn.keras`).
4. `streamlit run app.py`

## Results
| Model | Test accuracy |
|---|---|
| Dense (first version) | 36% |
| Dense (fixed) | ... |
| CNN + EarlyStopping | ... |

## Mistakes I fixed
- I used the test data as validation, so now I use `validation_split=0.2`
- I loaded gray images as rgb, so now I use `grayscale`
- I trained the same model twice, so now each model is trained once
- I added precision/recall, the loss plot and a CNN

## Limitations
- Weak with bad lighting, side faces or covered faces
- disgust has very few images
- The dataset may not represent everyone equally
