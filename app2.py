import os
import gradio as gr
from dotenv import load_dotenv
import datetime
from pathlib import Path

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Import your existing functions
from brain import encode_image, analyze_image_with_query
from patient_voice import record_audio, transcribe_with_groq
from ai_voice import text_to_speech_with_gtts

# System prompt
SYSTEM_PROMPT = """
You are a professional doctor (for educational purposes). Analyze what's in this image medically. 
If you find anything concerning, suggest potential remedies. 

Response guidelines:
- Format as if speaking directly to a patient
- Begin immediately with your assessment (no "In the image I see...")
- Keep concise (2-3 sentences max)
- No numbering or special characters
- Use natural doctor-patient language
Example: "With what I see, I think you may have... I recommend..."
"""

# Create output directory if it doesn't exist
OUTPUT_DIR = "doctor_responses"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def save_doctor_response(text_response, audio_path):
    """Save doctor's response to a text file and preserve the audio file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text response
    text_filename = f"{OUTPUT_DIR}/doctor_response_{timestamp}.txt"
    with open(text_filename, "w") as f:
        f.write(text_response)
    
    # Save audio file with new name
    if audio_path and os.path.exists(audio_path):
        new_audio_path = f"{OUTPUT_DIR}/doctor_voice_{timestamp}.mp3"
        os.rename(audio_path, new_audio_path)
        return new_audio_path
    
    return audio_path

def process_inputs(audio_filepath, image_filepath, save_response=False):
    try:
        speech_to_text_output = transcribe_with_groq(
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )

        # Handle the image input
        if image_filepath:
            doctor_response = analyze_image_with_query(
                query=SYSTEM_PROMPT + speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            doctor_response = "Please provide an image for medical analysis."

        voice_of_doctor = text_to_speech_with_gtts(
            input_text=doctor_response,
            output_filepath="final.mp3"
        )

        # Save response if requested
        if save_response:
            voice_of_doctor = save_doctor_response(doctor_response, voice_of_doctor)

        return speech_to_text_output, doctor_response, voice_of_doctor
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        return error_msg, error_msg, None

# Custom CSS
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #2d3748;
    text-align: center;
}
.description {
    text-align: center;
    color: #4a5568;
    margin-bottom: 20px;
}
.output-label {
    font-weight: bold;
    color: #2d3748;
}
.audio-container {
    margin-top: 15px;
}
.error-message {
    color: #e53e3e;
    font-weight: bold;
}
.save-container {
    margin-top: 20px;
    padding: 15px;
    background: #f7fafc;
    border-radius: 8px;
}
"""

def find_available_port(start_port=7860, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port

# Create absolute paths to example files
def get_example_path(filename):
    return os.path.join(os.path.dirname(__file__), "examples", filename)

# Create the interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🩺 AI Doctor with Vision and Voice")
    gr.Markdown("Describe your symptoms while showing any affected area. Our AI doctor will analyze and respond.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Your Symptoms",
                waveform_options={"sample_rate": 16000}
            )
            image_input = gr.Image(
                type="filepath",
                label="Upload Medical Image",
                elem_id="image-upload"
            )
            with gr.Row():
                submit_btn = gr.Button("Consult AI Doctor", variant="primary")
                save_checkbox = gr.Checkbox(
                    label="Save this consultation",
                    value=False,
                    info="Check to save doctor's response"
                )
        
        with gr.Column():
            speech_output = gr.Textbox(
                label="Your Description",
                interactive=False,
                lines=3
            )
            doctor_output = gr.Textbox(
                label="Doctor's Analysis",
                interactive=False,
                lines=4
            )
            audio_output = gr.Audio(
                label="Doctor's Voice Response",
                interactive=False,
                elem_classes="audio-container"
            )
            
            # Add download buttons
            with gr.Group(visible=False) as save_group:
                with gr.Row():
                    download_text_btn = gr.Button("Download Text Response")
                    download_audio_btn = gr.Button("Download Audio Response")
                download_text_file = gr.File(visible=False)
                download_audio_file = gr.File(visible=False)
            
            # Show saved confirmation
            save_confirmation = gr.Markdown("", visible=False)
    
    # Only add examples if the files exist
    example_files = [
        ["cough.wav", "skin_rash.jpg"],
        ["headache.wav", "eye_redness.jpg"]
    ]
    
    valid_examples = []
    for pair in example_files:
        audio_path = get_example_path(pair[0])
        image_path = get_example_path(pair[1])
        if os.path.exists(audio_path) and os.path.exists(image_path):
            valid_examples.append([audio_path, image_path])
    
    if valid_examples:
        gr.Examples(
            examples=valid_examples,
            inputs=[audio_input, image_input],
            outputs=[speech_output, doctor_output, audio_output],
            fn=process_inputs,
            cache_examples=False,
            label="Try Example Consultations"
        )
    
    # Show/hide download buttons based on response
    def toggle_download_buttons(doctor_response):
        if doctor_response and not doctor_response.startswith("An error occurred"):
            return gr.Group(visible=True)
        return gr.Group(visible=False)
    
    # Process inputs when submit button is clicked
    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, save_checkbox],
        outputs=[speech_output, doctor_output, audio_output]
    ).then(
        fn=toggle_download_buttons,
        inputs=doctor_output,
        outputs=save_group
    )
    
    # Handle download buttons
    def download_text(doctor_response):
        if doctor_response and not doctor_response.startswith("An error occurred"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"doctor_response_{timestamp}.txt"
            # Create temporary file
            temp_file = f"{OUTPUT_DIR}/{filename}"
            with open(temp_file, "w") as f:
                f.write(doctor_response)
            return gr.File(value=temp_file, visible=True, label=filename)
        return gr.File(visible=False)
    
    def download_audio(audio_file):
        if audio_file and os.path.exists(audio_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"doctor_voice_{timestamp}.mp3"
            return gr.File(value=audio_file, visible=True, label=filename)
        return gr.File(visible=False)
    
    # Connect download buttons to their functions
    download_text_btn.click(
        fn=download_text,
        inputs=doctor_output,
        outputs=download_text_file
    )
    
    download_audio_btn.click(
        fn=download_audio,
        inputs=audio_output,
        outputs=download_audio_file
    )

# Find an available port
available_port = find_available_port()

# Launch with specific settings
iface.launch(
    debug=True,
    share=False
)