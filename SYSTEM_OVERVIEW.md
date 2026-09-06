# System Overview: Multimodal Heart Disease Diagnostic Pipeline

## 1. Introduction
This system is an end-to-end multimodal diagnostic pipeline designed to assist clinicians in evaluating heart disease. It integrates specialized AI models to process diverse data types—visual (ECG), tabular (clinical metrics), and textual (reasoning)—to generate a comprehensive, explainable medical evaluation report.

## 2. System Architecture
The pipeline follows a "Specialized-to-Generalist" flow, where narrow domain experts (Upstream Models) extract features that are then synthesized by a clinical reasoning engine (Downstream Model).

### Upstream Specialized Models
1.  **GemmaECG-Vision**: A multimodal model that processes 12-lead ECG images. It translates visual waveform patterns into descriptive clinical findings.
2.  **Google TabFM**: A tabular-specialized model that analyzes clinical, demographic, and metabolic metrics to calculate a risk probability and a classification label (e.g., "High Risk").
3.  **Bio_ClinicalBERT**: Provides optimized text representations for clinical features to enhance the semantic understanding of the patient's medical history.

### Downstream Synthesis Engine
*   **BioMistral-7B**: A specialized clinical LLM that acts as the "Chief Resident." It ingests the outputs from the upstream models and the raw patient data to perform holistic reasoning and generate the final diagnostic report.

---

## 3. How the Pipeline Works (Workflow)

The system processes a patient case through the following sequential stages:

### Stage 1: Visual Feature Extraction
The user uploads a **12-lead ECG image**. This image is passed to **GemmaECG-Vision**, which identifies visual anomalies (like ST-segment depression or T-wave inversion) and outputs a text-based summary of the visual findings.

### Stage 2: Tabular Risk Assessment
The system ingests **23 patient metrics** (Age, BP, Cholesterol, BMI, etc.). **Google TabFM** processes these numbers to determine the statistical probability of heart disease and assigns a risk classification.

### Stage 3: Clinical Synthesis & Reasoning
All gathered information is compiled into a structured prompt for **BioMistral-7B**:
*   **Inputs**: Raw Clinical Record + GemmaECG-Vision Findings + TabFM Risk Score.
*   **Reasoning**: The model correlates the visual findings with the metabolic labs (e.g., checking if the ECG anomalies match the patient's cholesterol and blood pressure levels).
*   **Concurrence**: The model explicitly evaluates whether it agrees with the TabFM classification based on the holistic evidence.

### Stage 4: Final Report Generation
The system outputs a report divided into two sections:
1.  **Clinical Report & Hospital Analysis**: Technical analysis for medical professionals.
2.  **Patient-Friendly Summary**: A non-jargon explanation for the patient.

---

## 4. Endpoint Documentation

### Endpoint: `/predict`
**Method**: `POST`  
**Content-Type**: `multipart/form-data`

#### Request Parameters
1.  **`file` (File)**: An image file of the 12-lead ECG.
2.  **`metrics` (JSON)**: A set of 21 patient parameters.

| Category | Parameter | Type | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Demographics** | `age` | int | `58` | Patient's age in years |
| | `sex` | str | `"Male"` | Biological sex |
| | `ethnicity` | str | `"Caucasian"` | Patient's ethnicity |
| **Vitals & Labs** | `blood_pressure` | str | `"140/90"` | Systolic/Diastolic blood pressure |
| | `heart_rate` | int | `75` | Resting heart rate (bpm) |
| | `cholesterol` | int | `240` | Total cholesterol (mg/dL) |
| | `fasting_blood_sugar` | int | `110` | Fasting glucose levels (mg/dL) |
| | `diabetes` | str | `"Yes"` | Diabetes diagnosis status |
| | `bmi` | float | `28.5` | Body Mass Index (kg/m²) |
| **Cardiac Metrics** | `resting_ecg` | str | `"Normal"` | Result of resting ECG |
| | `oldpeak` | float | `1.5` | ST depression induced by exercise |
| | `slope` | str | `"Flat"` | Slope of peak exercise ST segment |
| | `major_vessels` | int | `2` | Number of major vessels colored by fluoroscopy |
| | `thalassemia` | str | `"Fixed"` | Thalassemia blood flow result |
| **Lifestyle** | `smoking` | str | `"Former"` | Smoking status (Never/Current/Former) |
| | `alcohol_consumption` | str | `"Moderate"` | Level of alcohol intake |
| | `physical_activity` | str | `"Low"` | Daily physical activity level |
| | `sleep_duration` | float | `6.5` | Average sleep hours per day |
| | `diet_quality` | str | `"Poor"` | Quality of dietary habits |
| | `stress_level` | str | `"High"` | Self-reported stress level |
| | `sedentary_lifestyle` | str | `"Yes"` | Whether the patient is primarily sedentary |

#### Response Format
```json
{
  "clinical_report": "Full Markdown formatted medical report...",
  "upstream_findings": {
    "gemma_ecg": "Visual findings from the ECG model...",
    "tabfm_risk": 75.5,
    "tabfm_label": "High Risk",
    "engineered_metrics": {
      "bmi_bp_interaction": 1.24,
      "health_risk_score": 0.78
    }
  }
}
```

---

## 5. Technical Optimizations
*   **Lazy Loading**: To reduce startup time and memory overhead, models are not loaded when the server starts. Instead, they are loaded into memory only upon the first request to the `/predict` endpoint.
*   **Memory Efficiency**: Models are loaded using `torch.float16` (half-precision) and `device_map="auto"` to optimize GPU/CPU utilization.
*   **Containerization**: The system is fully Dockerized for consistent deployment across GitHub (for code/CI) and Hugging Face Spaces (for hosting).
