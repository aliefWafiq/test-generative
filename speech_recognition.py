import soundfile as sf
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np

# Fungsi untuk membaca file audio
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = sf.read(path)
    return speech_array, sampling_rate

# Path ke file audio (ubah sesuai dengan file yang digunakan)
audio_path = "Recording.wav"

# Membaca file audio
speech_array, sampling_rate = speech_file_to_array_fn(audio_path)

# Konversi ke 16,000 Hz jika perlu
target_sample_rate = 16000
if sampling_rate != target_sample_rate:
    print(f"Mengonversi sample rate dari {sampling_rate} Hz ke {target_sample_rate} Hz...")
    speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sample_rate)
    sampling_rate = target_sample_rate  # Update nilai sample rate

# Load processor dan model Wav2Vec2 dari Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Konversi audio menjadi format tensor
input_values = processor(speech_array, return_tensors="pt", sampling_rate=sampling_rate).input_values

# Pastikan model dalam mode evaluasi
model.eval()

# Prediksi transkripsi
with torch.no_grad():
    logits = model(input_values).logits

# Dapatkan hasil teks dari prediksi
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("Hasil Transkripsi:", transcription)
