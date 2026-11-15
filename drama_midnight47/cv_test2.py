import sys, os, time, torch, soundfile as sf, numpy as np
sys.path.insert(0, "/ssd2/models/cosyvoice/CosyVoice")
sys.path.insert(0, "/ssd2/models/cosyvoice/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

M = "/ssd2/models/cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B"
cv = CosyVoice2(M, load_jit=False, load_trt=False, fp16=False)
print("[init] ok", flush=True)

PROMPT_TEXT = "这条路我走了很久，今天，终于轮到我来定规矩。"
refs = {
 "M46": "/ssd2/duanju/voices/samples_ext/M46_王大川_voice.wav",
 "F25": "/ssd2/duanju/voices/samples/F25_邱以沫_voice.wav",
}
texts = {
 "M46": "这一单，送不到。",
 "F25": "四十七单，目的地是拆迁楼。",
}
outdir="/ssd2/duanju/drama_midnight47/cv_test"
os.makedirs(outdir, exist_ok=True)
for cid, ref in refs.items():
    prompt = ref  # 直接传路径
    t1=time.time()
    n=0
    for j,out in enumerate(cv.inference_zero_shot(texts[cid], PROMPT_TEXT, prompt, stream=False)):
        w=out["tts_speech"]; w=w.detach().cpu().numpy().astype("float32").reshape(-1) if hasattr(w,"detach") else np.asarray(w,dtype="float32").reshape(-1)
        sf.write(f"{outdir}/{cid}_{j}.wav", w, 22050)
        n+=1
    d=sf.read(f"{outdir}/{cid}_0.wav")[0]
    print(f"[{cid}] {time.time()-t1:.2f}s chunks={n} dur={len(d)/22050:.2f}s peak={np.abs(d).max():.3f}", flush=True)
print("[done]")
