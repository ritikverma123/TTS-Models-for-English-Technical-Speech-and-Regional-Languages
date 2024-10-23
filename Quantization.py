from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from torch.quantization import quantize_dynamic

# Load the processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Example input
inputs = processor(text="OOP is centered around four main concepts: encapsulation, inheritance, polymorphism, and abstraction. Encapsulation involves bundling data and methods that operate on that data within a single unit or class, enhancing security and modularity. Inheritance allows a new class to inherit properties and behaviors from an existing class, promoting code reuse and reducing redundancy.", return_tensors="pt")

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Generate speech
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Save the generated speech to a WAV file
#sf.write("myenv/speech.wav", speech.numpy(), samplerate=16000)

# Quantize the model using Post-Training Quantization
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_speecht5_model.pth")
print("Quantized model saved.")
