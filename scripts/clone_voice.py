import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

ckpt_converter = 'OpenVoice/checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

print("Chargement du convertisseur de timbre...")
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

print("Extraction de l'empreinte vocale de référence (Jules Verne)...")
reference_speaker = 'reference_audio/ref_clip.wav'
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

text = "Bonjour, voici les prévisions météo pour demain. Un temps ensoleillé avec des températures autour de vingt degrés sur l'ensemble du pays."

print("Génération de la voix de base avec MeloTTS (français)...")
model = TTS(language='FR', device=device)
speaker_ids = model.hps.data.spk2id
print("Speakers disponibles :", speaker_ids)
speaker_key = list(speaker_ids.keys())[0]

src_path = f'{output_dir}/tmp_base.wav'
model.tts_to_file(text, speaker_ids[speaker_key], src_path, speed=1.0)

se_dir = 'OpenVoice/checkpoints_v2/base_speakers/ses'
print("Fichiers d'empreintes disponibles :", os.listdir(se_dir))
se_filename = [f for f in os.listdir(se_dir) if speaker_key.lower().split('-')[0] in f.lower()][0]
source_se = torch.load(os.path.join(se_dir, se_filename), map_location=device)

print("Conversion du timbre vers la voix de référence...")
save_path = f'{output_dir}/output_cloned.wav'
tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
    message="@MeteoConsultPoC"
)

print("Terminé ! Fichier généré :", save_path)
