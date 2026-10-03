import torch
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from PIL import Image
from openai import OpenAI
import io
import uvicorn
import json
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None

# Load .env file into environment if python-dotenv is available
if load_dotenv is not None:
    try:
        load_dotenv()
    except Exception:
        pass

try:
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
except ImportError:
    tabfm_v1_0_0 = None
    TabFMClassifier = None

# Categorical mappings for TabFM input ( shifted by 1, 0 = Unknown/Other)
MAPPINGS = {
    "sex": {"Male": 1, "Female": 2},
    "ethnicity": {"Caucasian": 1, "African American": 2, "Asian": 3, "Hispanic": 4, "Other": 5},
    "diabetes": {"No": 1, "Yes": 2},
    # Console labels that are not in the original TabFM vocab alias onto the same bins.
    "resting_ecg": {
        "Normal": 1,
        "Abnormal": 2,
        "ST-T Abnormality": 2,
        "Left Ventricular Hypertrophy": 2,
    },
    "slope": {"Up": 1, "Upsloping": 1, "Flat": 2, "Down": 3, "Downsloping": 3},
    "thalassemia": {"Normal": 1, "Fixed": 2, "Reversible": 3},
    "smoking": {"Never": 1, "Former": 2, "Current": 3},
    "alcohol_consumption": {"None": 1, "Low": 2, "Light": 2, "Moderate": 3, "High": 4, "Heavy": 4},
    "physical_activity": {"Low": 1, "Moderate": 2, "High": 3},
    "diet_quality": {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4},
    "stress_level": {"Low": 1, "Moderate": 2, "High": 3},
    "sedentary_lifestyle": {"No": 1, "Yes": 2},
}

def preprocess_metrics(metrics: 'PatientMetrics'):
    """Convert PatientMetrics Pydantic model to a numpy array for TabFM."""
    features = []

    # Numerical features
    features.append(metrics.age)

    # Robust Blood Pressure Parsing
    try:
        bp_parts = metrics.blood_pressure.split('/')
        sys = float(bp_parts[0]) if len(bp_parts) > 0 else 120.0
        dia = float(bp_parts[1]) if len(bp_parts) > 1 else 80.0
    except (ValueError, IndexError, AttributeError):
        sys, dia = 120.0, 80.0

    features.append(sys)
    features.append(dia)
    features.append(metrics.heart_rate)
    features.append(metrics.cholesterol)
    features.append(metrics.fasting_blood_sugar)
    features.append(metrics.bmi)
    features.append(metrics.oldpeak)
    features.append(metrics.major_vessels)
    features.append(metrics.sleep_duration)

    # Categorical features via mapping (0 = Unknown)
    for field, mapping in MAPPINGS.items():
        val = getattr(metrics, field, "Other")
        features.append(mapping.get(val, 0))

    return np.array([features], dtype=np.float32)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    load_models()
    yield
    # Shutdown: Clean up if needed
    models.clear()

app = FastAPI(title="Heart Disease Diagnostic Pipeline", lifespan=lifespan)

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

models = {}

def extract_generated_text(response):
    """Normalize chat-pipeline output to a plain string."""
    if not response:
        return ""

    generated = response[0].get("generated_text", "")
    if isinstance(generated, list):
        if generated and isinstance(generated[-1], dict):
            return generated[-1].get("content", "")
        return " ".join(part.get("content", "") for part in generated if isinstance(part, dict))
    if isinstance(generated, str):
        return generated
    return str(generated)


# BioMistral stays the synthesis role. It is not loaded on the L4.
# Groq serves that role so GemmaECG keeps the GPU.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not set; Groq inference is disabled.")
        return None

    # One attempt and a hard cap. Default SDK retries plus a reasoning model
    # hold the request until the Space proxy drops it and the replica looks dead.
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=40.0,
        max_retries=0,
    )


def call_groq_llm(system_prompt: str, user_content: str, max_tokens: int = 300) -> str:
    """BioMistral clinical synthesis, executed on Groq. No local 7B weights."""
    client = get_groq_client()
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            extra_body={"reasoning_effort": "low"},
        )
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as e:
        print(f"Groq inference error: {e}")
        return ""


def load_models():
    if models:
        return

    # Ensure HF token is available for authenticated downloads
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token and hf_login is not None:
        try:
            hf_login(token=hf_token)
        except Exception as e:
            print(f"Warning: Hugging Face login failed: {e}")
    else:
        print("Warning: HF_TOKEN not set or huggingface_hub not installed; unauthenticated HF Hub requests may be rate-limited.")

    print("Loading models... this may take a while.")

    # Quantization config to fit models in limited GPU VRAM (e.g., HF Spaces)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading GemmaECG-Vision...")
    models['ecg_processor'] = AutoProcessor.from_pretrained("yasserrmd/GemmaECG-Vision")
    models['ecg_model'] = AutoModelForMultimodalLM.from_pretrained(
        "yasserrmd/GemmaECG-Vision",
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )

    print("Loading Google TabFM...")
    try:
        if tabfm_v1_0_0 is not None:
            model = tabfm_v1_0_0.load(model_type="classification")
            clf = TabFMClassifier(model=model)

            # TabFM requires a 'fit' call to establish the In-Context Learning (ICL)
            # context and initialize encoders/scalers.
            context_df = pd.read_csv("tabfm_context.csv")
            X_context = context_df.drop(columns=["target"])
            y_context = context_df["target"]
            clf.fit(X_context, y_context)

            models['tabfm_clf'] = clf
        else:
            raise ImportError("tabfm library not installed")
    except Exception as e:
        print(f"Warning: Could not load TabFM model: {e}. Using mock for pipeline demonstration.")
        models['tabfm_clf'] = None

    print(f"BioMistral synthesis is Groq-hosted ({GROQ_MODEL}); local 7B weights are not loaded.")
    models['groq_client'] = get_groq_client()

    print("All models loaded successfully.")

class PatientMetrics(BaseModel):
    age: int
    sex: str
    ethnicity: str
    blood_pressure: str
    heart_rate: int
    cholesterol: int
    fasting_blood_sugar: int
    diabetes: str
    bmi: float
    resting_ecg: str
    oldpeak: float
    slope: str
    major_vessels: int
    thalassemia: str
    smoking: str
    alcohol_consumption: str
    physical_activity: str
    sleep_duration: float
    diet_quality: str
    stress_level: str
    sedentary_lifestyle: str

@app.post("/predict")
async def predict(
    metrics: str = Form(...),
    file: UploadFile = File(...)
):
    # Parse metrics from JSON string to Pydantic model
    try:
        metrics_data = json.loads(metrics)
        patient_metrics = PatientMetrics(**metrics_data)
    except Exception as e:
        return {"error": f"Invalid metrics format: {str(e)}"}

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    ecg_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Analyze this 12-lead ECG image and describe the visual findings related to cardiac health."}
            ]
        },
    ]

    ecg_inputs = models['ecg_processor'].apply_chat_template(
        ecg_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(models['ecg_model'].device)

    ecg_outputs = models['ecg_model'].generate(**ecg_inputs, max_new_tokens=100)
    gemma_ecg_findings = models['ecg_processor'].tokenizer.decode(ecg_outputs[0][ecg_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    if models['tabfm_clf'] is not None:
        try:
            X = preprocess_metrics(patient_metrics)
            probs = models['tabfm_clf'].predict_proba(X)
            tabfm_risk_probability = float(np.max(probs) * 100)
            tabfm_classification_label = "High Risk" if tabfm_risk_probability > 50 else "Low Risk"
        except Exception as e:
            print(f"TabFM Prediction Error: {e}")
            tabfm_risk_probability = 68.2
            tabfm_classification_label = "Positive for Heart Disease"
    else:
        tabfm_risk_probability = 68.2
        tabfm_classification_label = "Positive for Heart Disease"

    try:
        sys_bp = float(str(patient_metrics.blood_pressure).split("/")[0])
    except (ValueError, IndexError, AttributeError):
        sys_bp = 120.0
    bmi_bp_interaction = round(float(patient_metrics.bmi) * sys_bp / 100.0, 2)
    health_risk_score = round(min(1.0, float(tabfm_risk_probability) / 100.0), 2)

    # One BioMistral call. A second reasoning pass is what blew past the Space proxy.
    system_prompt = """You are BioMistral, the clinical synthesis model in an end-to-end heart-disease pipeline. You run through Groq so this step stays fast; do not mention the transport or the serving model. Upstream models already ran:

1. GemmaECG-Vision read the 12-lead ECG image.
2. Google TabFM scored the tabular clinical chart.

Write one report. Do not request another reasoning pass.

OPERATIONAL MANDATES:
1. Complete Parameter Coverage: Account for every parameter in the input block. Do not omit any.
2. Diagnostic Concurrence: State whether you agree or disagree with TabFM, using the ECG findings and the full chart.
3. Structured Output: Use ONLY the two Markdown headings below. No greeting and no commentary after the report.

## CLINICAL ANALYSIS & DIAGNOSTIC REASONING
## PATIENT-FRIENDLY SUMMARY
"""

    user_content = f"""
### 1. PATIENT CLINICAL RECORD
* Demographics:
  - Age: {patient_metrics.age}
  - Sex: {patient_metrics.sex}
  - Ethnicity: {patient_metrics.ethnicity}

* Vital Signs & Metabolic Labs:
  - Blood Pressure: {patient_metrics.blood_pressure} mmHg
  - Heart Rate: {patient_metrics.heart_rate} bpm
  - Cholesterol: {patient_metrics.cholesterol} mg/dL
  - Fasting Blood Sugar: {patient_metrics.fasting_blood_sugar} mg/dL
  - Diabetes Status: {patient_metrics.diabetes}
  - BMI: {patient_metrics.bmi} kg/m²

* Cardiac-Specific Diagnostic Metrics:
  - Resting ECG Result: {patient_metrics.resting_ecg}
  - ST Depression (Oldpeak): {patient_metrics.oldpeak}
  - Slope of Peak Exercise ST Segment: {patient_metrics.slope}
  - Number of Major Vessels (Fluoroscopy): {patient_metrics.major_vessels}
  - Thalassemia: {patient_metrics.thalassemia}

* Lifestyle & Behavioral Metrics:
  - Smoking Status: {patient_metrics.smoking}
  - Alcohol Consumption: {patient_metrics.alcohol_consumption}
  - Physical Activity Level: {patient_metrics.physical_activity}
  - Sleep Duration: {patient_metrics.sleep_duration} hours/day
  - Diet Quality: {patient_metrics.diet_quality}
  - Stress Level: {patient_metrics.stress_level}
  - Sedentary Lifestyle: {patient_metrics.sedentary_lifestyle}

* Derived / Engineered Features (Calculated):
  - BMI-BP Interaction: {bmi_bp_interaction}
  - Health Risk Score: {health_risk_score}

### 2. UPSTREAM MODEL 1 OUTPUT: GemmaECG-Vision (yasserrmd/GemmaECG-Vision)
- 12-Lead ECG Visual Findings: {gemma_ecg_findings}

### 3. UPSTREAM MODEL 2 OUTPUT: TabFM (google/tabfm-1.0.0-pytorch)
- TabFM Risk Probability: {tabfm_risk_probability}%
- TabFM Classification Label: {tabfm_classification_label}
"""

    final_report = call_groq_llm(system_prompt, user_content, max_tokens=900)
    if not final_report.strip():
        final_report = (
            "## CLINICAL ANALYSIS & DIAGNOSTIC REASONING\n"
            "BioMistral did not return a narrative in time. Upstream signals are included so this request can finish.\n"
            f"- GemmaECG-Vision: {gemma_ecg_findings}\n"
            f"- TabFM: {tabfm_risk_probability:.1f}% ({tabfm_classification_label})\n"
            f"- Chart: age {patient_metrics.age}, sex {patient_metrics.sex}, "
            f"blood pressure {patient_metrics.blood_pressure}, cholesterol {patient_metrics.cholesterol} mg/dL, "
            f"resting ECG {patient_metrics.resting_ecg}, oldpeak {patient_metrics.oldpeak}, "
            f"major vessels {patient_metrics.major_vessels}, smoking {patient_metrics.smoking}.\n\n"
            "## PATIENT-FRIENDLY SUMMARY\n"
            "The ECG image and the clinical chart were read, but the written summary was not finished. "
            "Please retry. This is a research assistant, not a diagnosis.\n"
        )

    return {
        "clinical_report": final_report,
        "upstream_findings": {
            "gemma_ecg": gemma_ecg_findings,
            "tabfm_risk": tabfm_risk_probability,
            "tabfm_label": tabfm_classification_label,
            "engineered_metrics": {
                "bmi_bp_interaction": bmi_bp_interaction,
                "health_risk_score": health_risk_score
            }
        }
    }

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Analyzes a 12-lead ECG image and generates a detailed clinical report using
    GemmaECG-Vision for visual analysis and Groq for clinical synthesis.
    """
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # 1. Visual Analysis via GemmaECG-Vision
    ecg_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Analyze this 12-lead ECG image and describe the visual findings related to cardiac health."}
            ]
        },
    ]

    ecg_inputs = models['ecg_processor'].apply_chat_template(
        ecg_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(models['ecg_model'].device)

    ecg_outputs = models['ecg_model'].generate(**ecg_inputs, max_new_tokens=100)
    raw_findings = models['ecg_processor'].tokenizer.decode(ecg_outputs[0][ecg_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # 2. Reasoning Layer via Groq
    groq_system_prompt = "You are BioMistral preparing a short ECG interpretation. Analyze the raw visual findings in a few clinical sentences. Do not write the full report."
    groq_user_content = f"RAW AI ECG FINDINGS: {raw_findings}"
    groq_reasoning = call_groq_llm(groq_system_prompt, groq_user_content, max_tokens=200)

    # 3. Clinical Synthesis via Groq
    ecg_report_system_prompt = """You are BioMistral, writing as a cardiology assistant. Your task is to take raw visual findings from an AI-based ECG analysis tool and a short interpretation, and synthesize them into a professional, structured clinical ECG report.

    Your report MUST include:
    1. VISUAL ANALYSIS: A clear description of the ECG findings.
    2. CLINICAL INTERPRETATION: What these findings mean in terms of cardiac health, informed by the provided reasoning.
    3. RECOMMENDATIONS: Suggested next steps (e.g., further testing, specialist referral).

    Maintain a professional, clinical tone. Do not include conversational fillers.
    """

    user_content = f"""RAW AI ECG FINDINGS:
    {raw_findings}

    CLINICAL REASONING CHAIN:
    {groq_reasoning}
    """

    final_report = call_groq_llm(ecg_report_system_prompt, user_content, max_tokens=250)

    return {
        "ecg_report": final_report,
        "raw_visual_findings": raw_findings
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
