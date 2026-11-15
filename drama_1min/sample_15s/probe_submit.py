import sys, os, json
sys.path.insert(0, "/ssd2/duanju/drama_1min/sample_15s")
os.chdir("/ssd2/duanju/drama_1min/sample_15s")
from gen_frames_15s import (upload_image, post, build_workflow, build_prompt,
                             CAST_ROOT, SCENES)
from shots_15s import SHOTS

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8191
IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
s = SHOTS[IDX]
cid = s["char"]
co = s.get("co_char")
refs = ["%s/%s/%s_%s.png" % (CAST_ROOT, cid, cid, s["view"])]
if co:
    refs.append("%s/%s/%s_headshot.png" % (CAST_ROOT, co, co))
refs.append(SCENES[s["scene"]])

print("refs:")
for r in refs:
    print("   ", r, os.path.exists(r))

up = []
for r in refs:
    up.append(upload_image(PORT, r))
print("uploaded:", [u["name"] for u in up])

wf = build_workflow(refs, build_prompt(s), 881234 + s["id"] * 7919, s)
for i, slot in enumerate(["6", "7", "8"]):
    wf[slot]["inputs"]["image"] = up[i]["name"] if i < len(up) else up[-1]["name"]

r = post("http://127.0.0.1:%d/prompt" % PORT, {"prompt": wf, "client_id": "probe15s"})
print("submit:", json.dumps(r)[:800])
