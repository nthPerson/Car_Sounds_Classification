# Introduction — Report Assembly

## Problem Context

- **Automotive diagnostics:** Vehicle sounds carry diagnostic information — experienced mechanics can identify faults by listening. Automating this with ML could democratize diagnostics.
- **Edge deployment imperative:** Real-time, on-device classification avoids cloud latency, works offline, preserves privacy (audio processed locally, never transmitted)
- **TinyML constraints:** Microcontrollers have severe resource limits (1 MB flash, 256 KB SRAM, no GPU) that challenge traditional ML deployment
- **Gap in literature:** Most audio classification research assumes server-side inference with GPUs. MCU deployment of audio classifiers — including on-device feature extraction and quantization — is underexplored.

## Research Questions

1. **Can a quantized neural network classify car diagnostic sounds on a Cortex-M4 microcontroller?** (Feasibility — does it fit? Does it run?)
2. **What is the accuracy cost of int8 quantization for small audio classification models?** (Quantization impact — PTQ vs QAT, statistical significance)
3. **How does the speaker-microphone domain gap affect real-world deployment accuracy?** (End-to-end evaluation — PC accuracy vs on-device playback accuracy)
4. **Does real-world noise augmentation improve model robustness for deployment?** (Data augmentation — synthetic vs noise-injected training)

## Hardware Target

| Property | Value |
|----------|-------|
| Board | Arduino Nano 33 BLE Sense Rev2 |
| MCU | nRF52840 (Cortex-M4F @ 64 MHz) |
| Flash | 1,024 KB (983 KB usable) |
| SRAM | 256 KB |
| Microphone | MP34DT06JTR PDM MEMS (on-board) |
| ADC | Built-in PDM interface, 16 kHz sample rate |
| Budget (model flash) | <= 150 KB |
| Budget (tensor arena) | <= 80 KB |

## Evaluation Framework

Four evaluation dimensions (from `docs/project_spec/evaluation_plan.md`):

| Dimension | What It Measures | Where |
|-----------|-----------------|-------|
| D1 | Classification accuracy (PC) | Keras/TFLite interpreter |
| D2 | Quantization impact | PC: float32 vs int8 |
| D3 | On-device performance | Arduino: latency, memory |
| D4 | End-to-end system | Arduino + speaker playback |

## Why This Matters (for Introduction Framing)

- Automotive industry moving toward predictive maintenance and connected vehicles
- Cost-effective edge devices could supplement or replace expensive diagnostic equipment
- Privacy-preserving: all inference happens on-device, no audio leaves the device
- BDA 602 contribution: end-to-end pipeline from raw audio to deployed MCU classifier with quantitative deployment evaluation

## Paper Structure Overview

1. Introduction (this section)
2. Data Sources — dataset, preprocessing, augmentation, noise collection
3. Methodology — models, quantization, deployment, evaluation framework
4. Analysis — interpretation of experimental observations
5. Results — quantitative tables and figures
6. Discussion & Industry Applications — implications, deployment recommendations
7. Conclusion & Future Work — answers to research questions, limitations, next steps

## Source Files
- `docs/project_spec/project_overview.md` — Project goals, scope, constraints
- `docs/project_spec/evaluation_plan.md` — Evaluation framework (D1-D4)
- `docs/project_spec/quantization_deployment_spec.md` — Hardware feasibility and deployment specification
- `docs/project_spec/development_roadmap.md` — 8-phase plan
