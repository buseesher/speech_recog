# 🎤 Turkish Digit Speech Recognition (HMM-Based)

This project performs speech recognition for Turkish digits (1 through 10) using Hidden Markov Models (HMM). 

It is implemented in Python and uses libraries such as `hmmlearn`, `python_speech_features`, and `pyaudio`.

## 📁 Project Structure

```bash
.
├── hmm/ # Trained HMM models (pickle files)
├── testingwav/ # Test audio files (.wav)
│ └── trainingwav/ # Training audio files
├── .idea/ # IDE configuration files
├── onlinemic.py # Real-time recognition using microphone input
├── testing.py # Recognition using test audio files
├── training.py # Training script to generate HMM models
  ```

## 🔧 Installation

```bash
pip install hmmlearn
pip install python_speech_features
pip install pyaudio
  ```
⚠️ Note: If you encounter issues installing pyaudio, make sure you have appropriate system dependencies installed (e.g., portaudio for Linux/macOS).

## ✅ Dependencies

`hmmlearn`

`python_speech_features`

`scipy`

`numpy`

`pyaudio`

## 📌 Notes

All digit audio files should follow the naming format: trainingwav_X_YY.wav where X is the digit (1–10) and YY is the sample number.

Make sure models are trained (training.py) before running any tests or real-time recognition.

