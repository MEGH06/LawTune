"""Central configuration for LawTune. Edit this file, not the scripts."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("LAWTUNE_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW = DATA / "raw"              # your Kaggle law JSONs go here
INTERIM = DATA / "interim"      # sampled sangraha, translations
PROCESSED = DATA / "processed"  # final train-ready datasets
OUT = ROOT / "outputs"
CPT_ADAPTER = OUT / "cpt_adapter"     # stage A LoRA
CPT_FUSED = OUT / "cpt_fused"         # stage A baked in; stage B's base
SFT_ADAPTER = OUT / "lawtune_adapter"  # stage B LoRA -- the thing you ship
FUSED = OUT / "lawtune_fused"          # stage B baked in

for _p in (RAW, INTERIM, PROCESSED, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- model
# We fine-tune the *instruct* checkpoint: a 2B base model cannot be taught chat
# formatting AND 22 languages AND law from scratch on a laptop. Starting from -it
# means we only have to bend it, not build it.
#
# Pre-quantized 4-bit weights, ~1.5 GB -- the whole reason this fits on a Mac.
# The base stays frozen and quantized; only LoRA adapters train. That is QLoRA,
# with MLX standing in for bitsandbytes (which is CUDA-only).
BASE_MODEL = os.environ.get("LAWTUNE_BASE", "mlx-community/gemma-2-2b-it-4bit")

# 1024, not 2048: attention cost is quadratic and unified memory is shared with
# everything else you have open. Raise it on an M-series Max/Ultra.
MAX_SEQ_LEN = int(os.environ.get("LAWTUNE_SEQ_LEN", 1024))
SEED = 3407

# ---------------------------------------------------------------- languages
# 22 scheduled languages of the Indian Constitution + English.
# key = ai4bharat/sangraha directory code
# val = (english name, native name, IndicTrans2 FLORES code, unicode script)
LANGUAGES = {
    "eng": ("English",    "English",   "eng_Latn", "Latin"),
    "asm": ("Assamese",   "অসমীয়া",     "asm_Beng", "Bengali"),
    "ben": ("Bengali",    "বাংলা",       "ben_Beng", "Bengali"),
    "brx": ("Bodo",       "बड़ो",        "brx_Deva", "Devanagari"),
    "doi": ("Dogri",      "डोगरी",       "doi_Deva", "Devanagari"),
    "gom": ("Konkani",    "कोंकणी",      "gom_Deva", "Devanagari"),
    "guj": ("Gujarati",   "ગુજરાતી",     "guj_Gujr", "Gujarati"),
    "hin": ("Hindi",      "हिन्दी",       "hin_Deva", "Devanagari"),
    "kan": ("Kannada",    "ಕನ್ನಡ",       "kan_Knda", "Kannada"),
    "kas": ("Kashmiri",   "کٲشُر",       "kas_Arab", "Arabic"),
    "mai": ("Maithili",   "मैथिली",      "mai_Deva", "Devanagari"),
    "mal": ("Malayalam",  "മലയാളം",     "mal_Mlym", "Malayalam"),
    "mar": ("Marathi",    "मराठी",       "mar_Deva", "Devanagari"),
    "mni": ("Manipuri",   "ꯃꯤꯇꯩꯂꯣꯟ",    "mni_Mtei", "Meetei"),
    "nep": ("Nepali",     "नेपाली",      "npi_Deva", "Devanagari"),
    "ori": ("Odia",       "ଓଡ଼ିଆ",       "ory_Orya", "Oriya"),
    "pan": ("Punjabi",    "ਪੰਜਾਬੀ",      "pan_Guru", "Gurmukhi"),
    "san": ("Sanskrit",   "संस्कृतम्",    "san_Deva", "Devanagari"),
    "sat": ("Santali",    "ᱥᱟᱱᱛᱟᱲᱤ",     "sat_Olck", "Ol_Chiki"),
    "snd": ("Sindhi",     "سنڌي",        "snd_Arab", "Arabic"),
    "tam": ("Tamil",      "தமிழ்",       "tam_Taml", "Tamil"),
    "tel": ("Telugu",     "తెలుగు",      "tel_Telu", "Telugu"),
    "urd": ("Urdu",       "اردو",        "urd_Arab", "Arabic"),
}
INDIC_LANGS = [k for k in LANGUAGES if k != "eng"]   # 22
ALL_LANGS = list(LANGUAGES)                          # 23

# ---------------------------------------------------------------- sampling
# Total UTF-8 megabytes of Sangraha to keep. Split evenly per language, so a
# low-resource language is never drowned by Hindi/Bengali. 200 MB is the ask.
SANGRAHA_TOTAL_MB = int(os.environ.get("SANGRAHA_TOTAL_MB", 200))
SANGRAHA_MIN_DOC_CHARS = 300
SANGRAHA_MAX_DOC_CHARS = 8000
SANGRAHA_SHARDS_PER_LANG = 3     # how many random parquet shards to stream from
SANGRAHA_STREAM_CAP = 120_000    # hard stop on rows scanned per language

# ---------------------------------------------------------------- translation
# Distilled 200M by default: the 1B is better but ~4x slower, which on a laptop is
# the difference between an afternoon and two days. Override with LAWTUNE_MT.
#   ai4bharat/indictrans2-en-indic-dist-200M   <- default, fast
#   ai4bharat/indictrans2-en-indic-1B          <- higher quality, much slower
TRANSLATE_MODEL = os.environ.get("LAWTUNE_MT", "ai4bharat/indictrans2-en-indic-dist-200M")
# How many English law QA pairs to translate into EACH of the 22 languages.
# 1000 x 22 = 22k multilingual legal turns. Lower it if you are short on time.
TRANSLATE_PAIRS_PER_LANG = int(os.environ.get("TRANSLATE_PAIRS_PER_LANG", 1000))
TRANSLATE_BATCH = 64
TRANSLATE_BEAMS = 1              # greedy; beams=5 triples runtime for marginal gain
TRANSLATE_MAX_SRC_CHARS = 1400   # IndicTrans2 is sentence-level; we chunk

# ---------------------------------------------------------------- system prompt
# Baked into every training example. The model learns this persona AND its limits.
SYSTEM_PROMPT = (
    "You are LawTune, a legal information assistant for Indian law "
    "(Constitution of India, IPC/BNS, CrPC/BNSS, Evidence Act, and Supreme Court "
    "precedent). Rules you always follow:\n"
    "1. Answer in the SAME language the user wrote in.\n"
    "2. Cite the exact Article/Section you rely on. If you cannot name one, say so.\n"
    "3. If the question is outside Indian law, refuse briefly and say what you do cover.\n"
    "4. If you are not sure, say you are not sure. Never invent a section number, "
    "case name, citation, or year.\n"
    "5. You give legal information, not legal advice. Recommend a licensed advocate "
    "for anything case-specific."
)

# ---------------------------------------------------------------- training
# MLX has no separate embedding LR, so stage A trains at one conservative rate.
# Iteration-based rather than epoch-based: on a laptop you want a knob you can
# point at the clock, not at the dataset.
#
# rank=16 -> 20.8M trainable params (0.79% of the model), a 42 MB adapter that fits
# under GitHub's 50 MB warning. rank=32 doubles both for a small quality gain; use it
# only if evaluation says you need it, and push the adapter to the Hub, not to git.
CPT = dict(
    rank=16, scale=16.0, dropout=0.0,
    keys=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    # Embedding LoRA is opt-in (--train-embeddings): it helps the low-resource scripts
    # but not every mlx-lm version can fuse a LoRA'd embedding back into the base.
    embedding_keys=["model.embed_tokens"],
    learning_rate=3e-5, iters=3000, batch_size=1, grad_accum=8,
    warmup=60, weight_decay=0.01, save_every=250, seq_len=1024,
)
SFT = dict(
    rank=16, scale=16.0, dropout=0.0,
    keys=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    learning_rate=1e-4, iters=6000, batch_size=1, grad_accum=8,
    warmup=100, weight_decay=0.01, save_every=250, seq_len=1024,
)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
