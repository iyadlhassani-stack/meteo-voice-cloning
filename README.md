# Meteo Voice Cloning — PoC

Preuve de concept de clonage vocal réalisée dans le cadre d'une mission avec **METEO CONSULT (La Chaîne Météo)** : cloner la voix d'un présentateur pour générer des bulletins météo sans nécessiter systématiquement sa présence en studio (gain de temps et de coût pour les mises à jour fréquentes).

⚠️ Ce repo est un PoC technique. La voix utilisée ici en démonstration est un narrateur du domaine public (Jules Verne, lu par Zeckou, LibriVox), pas la voix réelle du présentateur — pour des raisons évidentes de droits et de confidentialité tant que l'accord définitif avec le présentateur n'est pas formalisé.

## Résultats

Deux échantillons dans `samples/` :
- `demo_voix_base.wav` — voix générique produite par le moteur TTS (MeloTTS), avant clonage
- `demo_voix_clonee.wav` — même texte, après application du clonage de timbre (OpenVoice V2)

Mesures obtenues sur ce test :
- Similarité de timbre (cosinus) : 0.83
- WER (Word Error Rate) : 9.5% en brut, mais ~0% une fois la ponctuation et les nombres normalisés (Whisper transcrit "vingt" en "20" — pas une vraie erreur de reconnaissance)

## Architecture du pipeline

Texte du bulletin météo
→ MeloTTS (text-to-speech, français)
→ Audio "voix générique"
→ OpenVoice V2 — ToneColorConverter (utilise l'empreinte vocale extraite de l'audio de référence)
→ Audio final dans la voix cible

Modèles utilisés (tous gratuits, licence commerciale libre) :
- OpenVoice V2 (https://github.com/myshell-ai/OpenVoice) — MIT — conversion de timbre vocal, zero-shot
- MeloTTS (https://github.com/myshell-ai/MeloTTS) — MIT — synthèse text-to-speech de base

## Installation

Nécessite Python 3.11 (Python 3.9/3.10 posent des problèmes de wheels manquants sur certaines dépendances — voir "Pièges rencontrés" ci-dessous).

pyenv install 3.11.9
pyenv local 3.11.9
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchaudio

git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e . --no-deps
cd ..
pip install -r requirements.txt
pip install faster-whisper whisper-timestamped
pip install "numpy<2" "setuptools<81"

pip install git+https://github.com/myshell-ai/MeloTTS.git
python3 -m unidic download

pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='OpenVoice/checkpoints_v2')"

### Pièges rencontrés pendant l'installation (et solutions)

- Failed to build 'av' (CompileError Cython) : OpenVoice épingle faster-whisper==0.9.0, qui exige une version de av sans wheel disponible aujourd'hui → installer OpenVoice avec --no-deps, puis installer manuellement des versions récentes de faster-whisper/whisper-timestamped
- Même erreur après passage à Python 3.11 : confirme que le souci vient du pin de version, pas de Python lui-même
- numpy en conflit (1.x vs 2.x selon les paquets) : gradio/librosa (vieux) exigent numpy 1.x, dtw-python (récent) exige 2.x → forcer numpy<2, dtw-python n'est pas utilisé dans notre pipeline
- ModuleNotFoundError: pkg_resources : les versions récentes de setuptools (80+) ont retiré pkg_resources, requis par une vieille version de librosa → pip install "setuptools<81"
- Téléchargement Hugging Face bloqué indéfiniment : le protocole de transfert "Xet" de HF peut ne pas fonctionner selon le réseau → export HF_HUB_DISABLE_XET=1 avant de relancer
- Lien direct S3 des checkpoints OpenVoice mort (NoSuchBucket) : ancien lien de téléchargement obsolète → utiliser huggingface_hub.snapshot_download sur myshell-ai/OpenVoiceV2

## Utilisation

python3 scripts/clone_voice.py
python3 scripts/evaluate_clone.py

Pour cloner une nouvelle voix, remplacer reference_audio/ref_clip.wav par un extrait propre (30 sec à 2 min, idéalement enregistrement studio, mono, sans bruit de fond ni réverbération).

## Limites actuelles et prochaines étapes

- Le clonage est testé sur une voix de substitution (domaine public), pas encore sur un vrai enregistrement du présentateur METEO CONSULT
- Une seule courte référence audio a été utilisée ; la qualité du clone s'améliore avec plus de données propres et cohérentes (même micro, même pièce, faible réverbération)
- Pas encore de pipeline de génération automatisée à partir de données météo structurées
- Watermarking / traçabilité de l'audio généré à envisager avant mise en production, pour des raisons éthiques et de transparence

## Licence

OpenVoice V2 et MeloTTS sont sous licence MIT (usage commercial libre). Le code de ce repo est fourni tel quel dans le cadre d'un PoC.
