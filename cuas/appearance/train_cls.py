import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torchvision import datasets, models

CLASSES = ["balloon", "bird", "drone", "none", "trash"]
SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LowRes:

    def __init__(self, p=0.5, lo=8, hi=64):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, im):
        if random.random() > self.p:
            return im
        s = random.randint(self.lo, self.hi)
        return im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BILINEAR)


class Blur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, im):
        return im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5))) if random.random() < self.p else im


def fixed_res(s):
    return T.Compose([T.Resize((SIZE, SIZE)), T.Lambda(lambda im: im.resize((s, s), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/cuas_data/crops"))
    ap.add_argument("--out", default="../appearance_out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval-only", action="store_true", help="학습 없이 best.pt 로 시험·ONNX 내보내기만")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_tf = T.Compose([
        T.RandomResizedCrop(SIZE, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        LowRes(0.5), Blur(0.2),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.RandomGrayscale(0.3),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    ev_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = datasets.ImageFolder(f"{a.data}/train", tr_tf)
    va = datasets.ImageFolder(f"{a.data}/val", ev_tf)
    assert tr.classes == CLASSES, tr.classes
    counts = np.bincount(tr.targets, minlength=len(CLASSES))
    w = 1.0 / counts[tr.targets]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr), replacement=True)
    dl_tr = torch.utils.data.DataLoader(tr, a.bs, sampler=sampler, num_workers=8, pin_memory=True)
    dl_va = torch.utils.data.DataLoader(va, 256, num_workers=8)

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=a.epochs, steps_per_epoch=len(dl_tr))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def evaluate(dl):
        net.eval()
        cm = np.zeros((len(CLASSES), len(CLASSES)), int)
        with torch.no_grad():
            for x, y in dl:
                p = net(x.to(dev)).argmax(1).cpu().numpy()
                for t_, p_ in zip(y.numpy(), p):
                    cm[t_, p_] += 1
        return cm

    best, t0 = -1, time.time()
    for ep in range(0 if a.eval_only else a.epochs):
        net.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = lossf(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y)
        cm = evaluate(dl_va)
        bal = float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1)))
        print(f"ep{ep + 1:2d} loss {tot / len(tr):.3f}  val 균형정확도 {bal:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if bal > best:
            best = bal
            torch.save(net.state_dict(), f"{a.out}/best.pt")

    net.load_state_dict(torch.load(f"{a.out}/best.pt"))
    report = {"classes": CLASSES, "val_best_balanced_acc": best if best >= 0 else None}
    te = datasets.ImageFolder(f"{a.data}/test", ev_tf, allow_empty=True)
    cm = evaluate(torch.utils.data.DataLoader(te, 256, num_workers=8))
    report["test_confusion"] = cm.tolist()
    report["test_per_class_recall"] = dict(zip(CLASSES, (np.diag(cm) / np.maximum(cm.sum(1), 1)).round(3).tolist()))
    has = cm.sum(1) > 0
    report["test_balanced_acc"] = float(np.mean((np.diag(cm) / np.maximum(cm.sum(1), 1))[has]))
    report["by_size_px"] = {}
    for s in [6, 8, 12, 16, 24, 32, 48, 64, 128]:
        tes = datasets.ImageFolder(f"{a.data}/test", fixed_res(s), allow_empty=True)
        c = evaluate(torch.utils.data.DataLoader(tes, 256, num_workers=8))
        rec = np.diag(c) / np.maximum(c.sum(1), 1)
        hs = c.sum(1) > 0
        report["by_size_px"][s] = {"balanced_acc": round(float(rec[hs].mean()), 3),
                                   **{k: v for k, v, h in zip(CLASSES, rec.round(3).tolist(), hs) if h}}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    json.dump(report, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)

    net.eval().cpu()
    torch.onnx.export(net, torch.zeros(1, 3, SIZE, SIZE), f"{a.out}/cuas_cls.onnx", input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    json.dump({"classes": CLASSES, "size": SIZE, "mean": MEAN, "std": STD, "rgb": True},
              open(f"{a.out}/cuas_cls.json", "w"), indent=1)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
