from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import time

# Load the processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Example input
inputs = processor(text="OOP is centered around four main concepts: encapsulation, inheritance, polymorphism, and abstraction.", return_tensors="pt")

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Generate speech before pruning
speech_before = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("myenv/speech_before.wav", speech_before.numpy(), samplerate=16000)

# Pruning the model
parameters_to_prune = (
    (model, 'encoder.layers.0.self_attn.q_proj'),
    (model, 'encoder.layers.0.self_attn.k_proj'),
    (model, 'encoder.layers.0.self_attn.v_proj'),
    (model, 'encoder.layers.0.ffn.fc1'),
    (model, 'encoder.layers.0.ffn.fc2'),
)

# Pruning configuration
import torch.nn.utils.prune as prune

# Apply global unstructured pruning
for module, name in parameters_to_prune:
    prune.l1_unstructured(getattr(module, name), amount=0.3)  # Prune 30% of the weights

# Generate speech after pruning
speech_after = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("myenv/speech_after.wav", speech_after.numpy(), samplerate=16000)

# Measure inference time before and after pruning
def measure_inference_time(model, inputs, speaker_embeddings):
    start_time = time.time()
    _ = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    end_time = time.time()
    return end_time - start_time

# Timing inference
inference_time_before = measure_inference_time(model, inputs, speaker_embeddings)
inference_time_after = measure_inference_time(model, inputs, speaker_embeddings)

print(f'Inference time before pruning: {inference_time_before:.4f} seconds')
print(f'Inference time after pruning: {inference_time_after:.4f} seconds')

# Save the pruned model
torch.save(model.state_dict(), "pruned_speecht5_model.pth")
print("Pruned model saved.")