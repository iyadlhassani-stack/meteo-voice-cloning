import torch
import jiwer
from faster_whisper import WhisperModel
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt_converter = 'OpenVoice/checkpoints_v2/converter'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

print("=== Similarité de timbre (référence vs généré) ===")
reference_se, _ = se_extractor.get_se('reference_audio/ref_clip.wav', tone_color_converter, vad=True)
generated_se, _ = se_extractor.get_se('outputs_v2/output_cloned.wav', tone_color_converter, vad=True)

ref_flat = reference_se.squeeze()
gen_flat = generated_se.squeeze()
cosine_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), gen_flat.unsqueeze(0)).item()
print(f"Similarité cosinus : {cosine_sim:.4f}  (proche de 1 = timbre très proche)")

print("\n=== Intelligibilité (WER) ===")
reference_text = "Bonjour, voici les prévisions météo pour demain. Un temps ensoleillé avec des températures autour de vingt degrés sur l'ensemble du pays."

model = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")
segments, info = model.transcribe("outputs_v2/output_cloned.wav", language="fr")
transcription = " ".join([seg.text for seg in segments]).strip()
print(f"Texte attendu     : {reference_text}")
print(f"Texte transcrit   : {transcription}")

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

ref_normalized = transform(reference_text)
hyp_normalized = transform(transcription)

print(f"Texte attendu (normalisé)  : {ref_normalized}")
print(f"Texte transcrit (normalisé): {hyp_normalized}")

wer = jiwer.wer(ref_normalized, hyp_normalized)

print(f"WER : {wer*100:.2f}%")
