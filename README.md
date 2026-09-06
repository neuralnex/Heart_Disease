# Multimodal Heart Disease Diagnostic Pipeline

An advanced, end-to-end clinical decision-support system that integrates multimodal AI models to evaluate heart disease risk. This pipeline combines visual analysis of ECGs, tabular clinical data processing, and large-scale medical reasoning to generate explainable diagnostic reports.

## 🚀 Overview

The pipeline employs a "Specialized-to-Generalist" architecture. It uses upstream domain-specific models to extract high-level features from diverse data sources, which are then synthesized by a clinical LLM to produce a final medical evaluation.

### Key Components
- **Visual Analysis**: `GemmaECG-Vision` for 12-lead ECG waveform interpretation.
- **Tabular Risk**: `Google TabFM` for metabolic and clinical metric risk scoring.
- **Clinical Embeddings**: `Bio_ClinicalBERT` for medical text representation.
- **Reasoning Engine**: `BioMistral-7B` for multimodal synthesis and report generation.

## ✨ Features

- **Multimodal Integration**: Seamlessly combines image (ECG) and tabular (Clinical) data.
- **Explainable AI**: Generates a dual-format report (one for clinicians, one for patients).
- **Diagnostic Concurrence**: Explicitly validates the agreement between statistical risk (TabFM) and holistic clinical reasoning (BioMistral).
- **Production-Ready**: Full Docker support with memory-optimized loading (float16) and lazy loading for efficient deployment.

## 🛠️ Installation & Deployment

### Prerequisites
- Docker
- NVIDIA GPU (recommended for inference speed)

### Local Deployment
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/Heart_Disease.git
   cd Heart_Disease
   ```
2. Build the Docker image:
   ```bash
   docker build -t heart-disease-diag .
   ```
3. Run the container:
   ```bash
   docker run -p 7860:7860 heart-disease-diag
   ```

### Hugging Face Spaces Deployment
The `/Heart` directory is optimized for Hugging Face Spaces. To deploy:
1. Upload the contents of the `Heart/` folder to a new Space.
2. Select the `Docker` SDK.

## 📖 Documentation

For a detailed explanation of the architecture, the 23 input metrics, and the endpoint specifications, please refer to the **[System Overview](./SYSTEM_OVERVIEW.md)**.

## 📡 API Usage

**Endpoint**: `POST /predict`

**Input**: 
- 12-lead ECG Image (File)
- 23 Patient Metrics (JSON)

**Output**:
- Comprehensive Clinical Report
- Patient-Friendly Summary
- Upstream model findings (ECG descriptions & TabFM risk %)

---
*Disclaimer: This system is designed for educational and research purposes and should be used as a decision-support tool, not as a replacement for professional medical diagnosis.*
