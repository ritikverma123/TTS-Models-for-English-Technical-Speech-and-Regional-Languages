from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import time
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic

# Step 1: Load Model and Processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Step 2: Prepare Input
inputs = processor(text="OOP is centered around four main concepts: encapsulation, inheritance, polymorphism, and abstraction.", return_tensors="pt")

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Step 3: Generate Speech Before Optimization
speech_before = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("myenv/speech_before.wav", speech_before.numpy(), samplerate=16000)

# Step 4: Model Quantization
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Step 5: Pruning the Model
# Example: Prune all linear layers in the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.18)  # Prune 30% of weights

model.eval()
quantized_model.eval()

# Step 6: Generate Speech After Optimization
speech_after = quantized_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("myenv/speech_after.wav", speech_after.numpy(), samplerate=16000)

# Step 7: Measure Inference Time
def measure_inference_time(model, inputs, speaker_embeddings):
    start_time = time.time()
    _ = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    end_time = time.time()
    return end_time - start_time

# Timing inference
inference_time_before = measure_inference_time(model, inputs, speaker_embeddings)
inference_time_after = measure_inference_time(quantized_model, inputs, speaker_embeddings)

print(f'Inference time before optimization: {inference_time_before:.4f} seconds')
print(f'Inference time after optimization: {inference_time_after:.4f} seconds')

# Step 8: Save the Optimized Model
torch.save(quantized_model.state_dict(), "optimized_speecht5_model.pth")
print("Optimized model saved.")
