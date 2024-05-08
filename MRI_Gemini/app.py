import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
generation_config = {
    "temperature" : 0.5,
    "top_p" : 1,
    "top_k" : 32,
    "max_output_tokens" : 4096,
}

# safety_settings = [
#     {"category" : "HARM_CATEGORY_"+category, "threshold" : "BLOCK_MEDIUM_AND_ABOVE"}
#     for category in ["HATE_SPEECH", "HARRASMENT", "DANGEROUS_CONTENT"]

# ]

model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    # safety_settings=safety_settings
)

def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find the image: {image_path}")
    
    return {"mime_type" : "image/jpeg" , "data":image_path.read_bytes()}

def generate_gemini_reply(prompt, image_path):
    image_data = read_image_data(image_path)
    response = model.generate_content([prompt,image_data])
    return response.text

input_prompt = """ **## MRI Analysis for Neurologist

**Your expertise as a neurologist is crucial in interpreting MRI scans and diagnosing neurological conditions. You will be provided with an MRI scan, and your role is to conduct a comprehensive analysis based on the following guidelines:**

**Analysis Guidelines:**

1. **Anatomical Evaluation:** Analyze the MRI images for any abnormalities in brain anatomy, including structural malformations, atrophy, or masses. 

2. **Signal Abnormalities:** Identify any deviations in signal intensity within the brain tissue, which could indicate inflammation, demyelination, ischemia, or other pathological processes.

3. **Lesion Characterization:** If lesions are present, describe their size, location, shape, and signal characteristics on different sequences (T1, T2, FLAIR) to aid in differential diagnosis.

4. **Correlation with Clinical Presentation:** Consider the patient's clinical history and symptoms while interpreting the MRI findings. This will help narrow down the possibilities and determine the most likely cause of the neurological presentation.

5. **Differential Diagnosis:** Based on the MRI findings and clinical presentation, propose a list of potential neurological conditions that could explain the patient's symptoms.

6. **Recommendations:**  Recommend further investigations if needed, such as additional MRI sequences, blood tests, or cerebrospinal fluid analysis, to confirm the diagnosis.  Outline potential treatment options based on the most likely diagnosis. 

**Important Note:**

* The MRI analysis should be interpreted in conjunction with the patient's clinical history and physical examination.
* This analysis does not constitute a definitive diagnosis and should be used for guidance in patient management.

**Disclaimer:**

* This interpretation is based on the provided MRI scan and should not be used for definitive medical decision-making. Consultation with the referring physician is essential for comprehensive patient care.

**By providing a thorough analysis of the MRI scan, you can significantly contribute to the accurate diagnosis and management of neurological conditions.**
"""


def process_upload_files(files):
    file_path = files[0].name if files else None
    response = generate_gemini_reply(input_prompt,file_path) if file_path else None
    return file_path, response

with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output,file_output]

    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types = ["image"],
        file_count = "multiple",
    )
    upload_button.upload(process_upload_files,upload_button,combined_output)

demo.launch(debug = True)
