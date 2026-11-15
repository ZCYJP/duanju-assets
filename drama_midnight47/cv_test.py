import sys, os, time, torch, soundfile as sf
sys.path.insert(0, "/ssd2/models/cosyvoice/CosyVoice")
sys.path.insert(0, "/ssd2/models/cosyvoice/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

M = "/ssd2/models/cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B"
print("[init] loading model...", flush=True)
t0=time.time()
cv = CosyVoice2(M, load_jit=False, load_trt=False, fp16=False)
print(f"[init] ok {time.time()-t0:.1f}s", flush=True)

ref = "/ssd2/duanju/voices/samples_ext/M46_王大川_voice.wav"
prompt = load_wav(ref, 16000)
print("[ref]", prompt.shape, prompt.dtype, flush=True)

texts = [
  "这一单，送不到。",
  "四十七单，目的地是拆迁楼。",
]
outdir="/ssd2/duanju/drama_midnight47/cv_test"
os.makedirs(outdir, exist_ok=True)
for i,t in enumerate(texts):
    t1=time.time()
    for j,out in enumerate(cv.inference_zero_shot(t, prompt, stream=False)):
        sf.write(f"{outdir}/cv_{i}_{j}.wav", out["tts_speech"], 22050)
    print(f"[gen {i}] {time.time()-t1:.2f}s -> {t}", flush=True)
print("[done]")
