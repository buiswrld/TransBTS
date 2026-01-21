import os, numpy as np, pickle

MODS = ["flair","ct1","t1","t2"]

def pkload(p):
    with open(p,"rb") as f: return pickle.load(f)

def audit(list_file, root):
    bad = []
    with open(list_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        path = os.path.join(root, ln, "data_f32b0.pkl")
        img, lbl = pkload(path)
        # img is (240,240,155,4)
        for c in range(img.shape[-1]):
            v = img[..., c]
            nz = (v != 0).mean()
            sd = v[v!=0].std() if (v!=0).any() else 0
            if nz < 0.01 or sd == 0 or np.isnan(v).any() or np.isinf(v).any():
                bad.append((ln, c, nz, sd, float(v.min()), float(v.max())))
                break

    print("flagged cases:", len(bad))
    for row in bad[:20]:
        print(row)

audit("/lambda/nfs/KAMS/TransBTS/train.txt", "/lambda/nfs/KAMS/Imaging")
