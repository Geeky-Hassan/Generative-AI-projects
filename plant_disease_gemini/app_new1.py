import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key using the loaded environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel with the specified model name, configuration, and safety settings
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Function to generate a response based on a prompt and an image path
def generate_gemini_response(prompt, image_path):
    image_data = read_image_data(image_path)
    response = model.generate_content([prompt, image_data])
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

## Dentist Use Case

input_prompt = """As a highly skilled dentist, your expertise is crucial in maintaining optimal oral health for your patients. You will be provided with information or symptoms related to dental problems, and your role involves conducting a thorough analysis to identify the specific issues, propose treatment plans, and offer preventive recommendations.

**Analysis Guidelines:**

1. **Diagnosis:** Examine the information provided by the patient (symptoms, medical history) or conduct a dental examination to accurately diagnose oral health problems.

2. **Detailed Findings:**  Provide a clear explanation of the identified dental issue,  including affected teeth/areas, symptoms, and potential causes.

3. **Treatment Plan:** Outline a personalized treatment plan to address the identified dental problem. This may involve procedures, medications, or further investigation (e.g., X-rays).

4. **Preventive Recommendations:** Offer comprehensive recommendations for maintaining good oral hygiene, preventing future dental problems, and optimizing overall oral health.  This may include brushing/flossing techniques, dietary advice, and regular dental checkups.

5. **Important Note:** As a dentist, your insights are vital for informed decision-making regarding oral health.  Your response should be thorough, concise, and focused on the patient's well-being.

**Disclaimer:**

*Please note that the information provided is based on a general dental analysis and should not replace professional dental advice. Consult with a qualified dentist before implementing any treatment plans.*

Your expertise plays a key role in ensuring the health and longevity of your patients' smiles.  Proceed to analyze the provided information or conduct a dental examination, adhering to the outlined structure.
"""

# input_prompt = """
# As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

# **Analysis Guidelines:**

# 1. **Disease Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

# 2. **Detailed Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

# 3. **Next Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

# 4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

# 5. **Important Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

# **Disclaimer:**
# *"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

# Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
# """
def process_uploaded_files(files):
    file_path = files[0].name if files else None
    response = generate_gemini_response(input_prompt, file_path) if file_path else None
    return file_path, response


with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    upload_button = gr.UploadButton(
        "Click to upload an image",
        file_types = ["image"],
        file_count = "multiple",
    )
    upload_button.upload(process_uploaded_files,upload_button,combined_output)

demo.launch(debug = True)
