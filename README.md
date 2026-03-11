# Car Sound Classification

A BDA 602 group project that trains a neural network to identify car engine problems by listening to audio — then runs it on a tiny microcontroller in real time.

## What This Project Does

1. **Listens** to short clips of car engine audio (1.5 seconds each)
2. **Classifies** the sound into one of several categories — normal operation, specific faults (belt issues, engine misfires, etc.), or combined problems
3. **Runs on a small device** (Arduino Nano 33 BLE Sense Rev2) that could eventually sit under a car's hood and flag issues as they happen

## Why It Matters

Diagnosing car problems by sound usually requires an experienced mechanic. This project explores whether a cheap, low-power microcontroller can do the same job automatically using machine learning.

## The Dataset

We use the [Car Diagnostics Dataset](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset) from Kaggle — 1,386 audio recordings across 13 different engine conditions (normal, belt problems, misfires, exhaust leaks, and more).

## How It Works (High Level)

- Audio clips are converted into visual-like representations called **spectrograms** that capture the frequency patterns in the sound
- A small convolutional neural network (CNN) learns to tell the difference between healthy and faulty engine sounds
- The trained model is **compressed** (quantized to int8) so it fits within the Arduino's very limited memory (256 KB of RAM, 1 MB of storage)
- On the device, the microphone captures audio, the model runs inference, and the result is reported in real time

## Getting Started

See [GETTING_STARTED.md](GETTING_STARTED.md) for setup instructions, how to run the code, and pointers to all project documentation.

## Repository Structure

```
Car_Sounds/
├── car_diagnostics_dataset/   # Raw audio files (downloaded from Kaggle)
├── data/                      # Processed features and train/val/test splits
├── docs/                      # Project specs, plans, and development log
├── src/                       # Python source modules
├── notebooks/                 # Jupyter notebooks for each development phase
├── arduino/                   # Arduino sketch for on-device deployment
├── models/                    # Trained model files
├── results/                   # Evaluation figures and tables
├── project_setup/             # Dataset verification and early exploration
├── GETTING_STARTED.md         # How to set up and run everything
└── CLAUDE.md                  # Instructions for AI-assisted development
```
