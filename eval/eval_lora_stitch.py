#!/usr/bin/env python
"""Few-shot stitch: make M5's pointer head use an external LoRA's representation space.

Everything is frozen except a handful of scalars fitted on <=30 labelled rows:
  A  steering with class directions taken from the LoRA's trained (alpaca) prompt space:
     v_c = mean(h_alp | c) - mean(h_alp), scored by head(h_sys + alpha * v_c)[c]
  A2 global prompt direction: v = mean(h_alp - h_sys)
  B  gated fusion (prompt key in the readout): logits = head(h_sys) + beta * log p_alp,
     with p_alp a nearest-class-mean probe in the LoRA's alpaca space.
Diagnostics (no training): NCM / logistic probes on h_alp and h_sys.

    python eval/eval_lora_stitch.py --run /mnt/f/distill_jev_runs/main \
        --ext "./github/Trained Models/My Dataset/gemma-4-e4b_claim" \
        --train /mnt/f/distill_jev_runs/prep/costbr-30 \
        --test /mnt/f/distill_jev_runs/prep/bench-costbr-test \
        --out /mnt/f/distill_jev_runs/stitch
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import numpy as np
import torch

from data.config import DELIMITERS, MODEL
from data.suites import write_json
from model.delimiters import apply_delimiter_deltas
from model.head import PointerHead
from model.load import compute_dtype, language_model

ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-prompts", required=True,
                    help='JSON: conv_key -> {"prompt": ...} for --train (the alpaca Instruction slot)')
    ap.add_argument("--test-prompts", required=True, help="same, for --test")
    ap.add_argument("--alpha-max", type=float, default=6.0)
    ap.add_argument("--alpha-step", type=float, default=0.25)
    return ap.parse_args()


def load_model(a):
    from peft import PeftModel
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=a.model, max_seq_length=4096, dtype=compute_dtype(), load_in_4bit=True,
        use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": DELIMITERS})
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
    deltas = apply_delimiter_deltas(model, language_model, delimiter_ids)
    model = PeftModel.from_pretrained(model, str(Path(a.run) / "adapter"))
    model.load_adapter(a.ext, adapter_name="ext", is_trainable=False)
    name = f"combo_w{str(a.weight).replace(chr(46), chr(112))}"
    model.add_weighted_adapter(["default", "ext"], [1.0, a.weight], name, combination_type="linear")
    model.set_adapter(name)
    saved = torch.load(Path(a.run) / "head.pt", map_location="cpu", weights_only=False)
    for key, tensor in saved["deltas"].items():
        deltas[key].data.copy_(tensor.to(deltas[key].device))
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size")
    head = PointerHead(hidden_size, dp=saved["head"]["q.weight"].shape[0],
                       garbage=bool(saved.get("garbage", False)))
    head.load_state_dict(saved["head"])
    head.temperature = 1.0
    model.head = head.to("cuda").eval()
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model, tokenizer, head


@torch.no_grad()
def sys_features(model, dataset, device):
    h_sys, h_opts, owner, ys = [], [], [], []
    for i, row in enumerate(dataset):
        ids = torch.tensor([row["input_ids"]], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False,
                       logits_to_keep=1).hidden_states[-1][0]
        h_sys.append(hidden[row["decide_pos"]].float().cpu())
        for j, p in enumerate(row["opt_pos"]):
            h_opts.append(hidden[p].float().cpu())
            owner.append(i)
        ys.append(int(row["label"]))
        if (i + 1) % 200 == 0:
            print(f"[feat] sys {i + 1}/{len(dataset)}", flush=True)
    return (torch.stack(h_sys), torch.stack(h_opts), torch.tensor(owner), torch.tensor(ys))


@torch.no_grad()
def alpaca_features(model, tokenizer, texts, device):
    out = []
    for i, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False,
                       logits_to_keep=1).hidden_states[-1][0]
        out.append(hidden[-1].float().cpu())
        if (i + 1) % 200 == 0:
            print(f"[feat] alpaca {i + 1}/{len(texts)}", flush=True)
    return torch.stack(out)


def conv_key(dataset, i):
    return dataset[i]["record_id"].split("/")[-1]


def alpaca_texts(dataset, prompts_file):
    prompts = json.load(open(prompts_file))
    texts = []
    for i in range(len(dataset)):
        key = conv_key(dataset, i)
        texts.append(ALPACA.format(prompts[key]["prompt"], "", ""))
    return texts


def head_option_logits(head, h_decide, h_opts, owner):
    dev = next(head.parameters()).device
    with torch.no_grad():
        return head.many(h_decide.to(dev), h_opts.to(dev), owner.to(dev)).float().cpu()


def ncm(h_train, y_train, h_test, n_classes):
    means = torch.stack([h_train[y_train == c].mean(0) for c in range(n_classes)])
    dist = torch.cdist(h_test, means)
    return dist, means


def check_label_is_class(*datasets):
    """Fail loudly if the option index is not a stable class identity.

    The NCM probes cluster on `row["label"]`, which is the *option index*. If the rows were
    prepped without --no-shuffle then data/prep_dataset.py shuffles the criteria per row, so the
    same class lands on a different index each time and every "class mean" is a mixture of all
    the classes. That does not look like a failure: on RumourEval it gave probe accuracy 0.068,
    i.e. far *below* the 0.25 chance rate, and was easy to read as "this LoRA has no linearly
    decodable signal" when the real problem was the prep.

    So require class -> index to be a bijection, which is what a fixed-label task needs.
    """
    seen = {}
    for ds in datasets:
        for row in ds:
            key = list(row["keys"])[int(row["label"])]
            seen.setdefault(key, set()).add(int(row["label"]))
    bad = {k: sorted(v) for k, v in seen.items() if len(v) != 1}
    if bad:
        raise SystemExit(
            "the option index is not a stable class identity, so the probe would mix classes:\n"
            f"  {bad}\n"
            "  Re-prep the dataset with --no-shuffle so the criteria keep their authored order.")
    return {k: sorted(v)[0] for k, v in seen.items()}


def class_means(h_train, y_train, n_classes):
    return torch.stack([h_train[y_train == c].mean(0) for c in range(n_classes)])


def acc(pred, y):
    return float((pred == y).float().mean())


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_from_disk

    train_ds, test_ds = load_from_disk(a.train), load_from_disk(a.test)
    label_map = check_label_is_class(train_ds, test_ds)
    model, tokenizer, head = load_model(a)
    device = next(model.parameters()).device

    cache = out / f"features_w{a.weight}.pt"
    if cache.exists():
        payload = torch.load(cache, weights_only=False)
        h_sys_tr, h_opt_tr, owner_tr, y_tr = payload["sys_tr"]
        h_sys_te, h_opt_te, owner_te, y_te = payload["sys_te"]
        h_alp_tr, h_alp_te = payload["alp_tr"], payload["alp_te"]
        print("[feat] loaded cache", flush=True)
    else:
        h_sys_tr, h_opt_tr, owner_tr, y_tr = sys_features(model, train_ds, device)
        h_sys_te, h_opt_te, owner_te, y_te = sys_features(model, test_ds, device)
        h_alp_tr = alpaca_features(model, tokenizer, alpaca_texts(train_ds, a.train_prompts), device)
        h_alp_te = alpaca_features(model, tokenizer, alpaca_texts(test_ds, a.test_prompts), device)
        torch.save({"sys_tr": (h_sys_tr, h_opt_tr, owner_tr, y_tr), "sys_te": (h_sys_te, h_opt_te, owner_te, y_te),
                    "alp_tr": h_alp_tr, "alp_te": h_alp_te}, cache)

    logits_tr = head_option_logits(head, h_sys_tr, h_opt_tr, owner_tr).reshape(len(train_ds), -1)
    logits_te = head_option_logits(head, h_sys_te, h_opt_te, owner_te).reshape(len(test_ds), -1)
    base_te = acc(logits_te.argmax(-1), y_te)

    report = {"weight": a.weight, "n_train": len(train_ds), "n_test": len(test_ds),
              "label_map": label_map,
              "pointer_base_test": base_te, "pointer_base_train": acc(logits_tr.argmax(-1), y_tr)}

    K = int(y_tr.max()) + 1
    if set(y_tr.tolist()) != set(range(K)) or set(y_te.tolist()) - set(range(K)):
        raise SystemExit(f"the few-shot pool must cover classes 0..{K - 1}; "
                         f"got train {sorted(set(y_tr.tolist()))} test {sorted(set(y_te.tolist()))}")
    report["n_classes"] = K
    d_alp, _ = ncm(h_alp_tr, y_tr, h_alp_te, K)
    d_sys, _ = ncm(h_sys_tr, y_tr, h_sys_te, K)
    report["ncm_alpaca_test"] = acc(d_alp.argmin(-1), y_te)
    report["ncm_sys_test"] = acc(d_sys.argmin(-1), y_te)

    from sklearn.linear_model import LogisticRegression
    for tag, htr, hte in (("logreg_alpaca", h_alp_tr, h_alp_te), ("logreg_sys", h_sys_tr, h_sys_te)):
        clf = LogisticRegression(max_iter=5000, C=0.05).fit(htr.numpy(), y_tr.numpy())
        report[f"{tag}_test"] = float((clf.predict(hte.numpy()) == y_te.numpy()).mean())
        report[f"{tag}_train"] = float((clf.predict(htr.numpy()) == y_tr.numpy()).mean())

    def steer_class(alphas):
        mu = h_alp_tr.mean(0)
        v = torch.stack([h_alp_tr[y_tr == c].mean(0) - mu for c in range(K)])
        scores = {}
        for alpha in alphas:
            for name, hs, yy in (("train", h_sys_tr, y_tr), ("test", h_sys_te, y_te)):
                sc = torch.zeros(len(hs), K)
                for c in range(K):
                    z = head_option_logits(head, hs + alpha * v[c], h_opt_tr if name == "train" else h_opt_te,
                                           owner_tr if name == "train" else owner_te).reshape(len(hs), -1)
                    sc[:, c] = z[:, c]
                scores.setdefault(name, {})[str(alpha)] = acc(sc.argmax(-1), yy)
        return scores

    alphas = [round(x * a.alpha_step, 3) for x in range(int(a.alpha_max / a.alpha_step) + 1)]
    steer = steer_class(alphas)
    best_alpha = max((v, k) for k, v in steer["train"].items())
    report["steer_class_alpha_test"] = steer["test"][best_alpha[1]]
    report["steer_class_alpha_train"] = steer["train"][best_alpha[1]]
    report["steer_class_best_alpha"] = float(best_alpha[1])
    report["steer_class_curves"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in steer.items()}

    mu_all = h_alp_tr.mean(0)
    v_global = mu_all - h_sys_tr.mean(0)
    global_curve = {}
    for alpha in alphas:
        for name, hs, ho, own, yy in (("train", h_sys_tr, h_opt_tr, owner_tr, y_tr),
                                       ("test", h_sys_te, h_opt_te, owner_te, y_te)):
            z = head_option_logits(head, hs + alpha * v_global, ho, own).reshape(len(hs), -1)
            global_curve.setdefault(name, {})[str(alpha)] = acc(z.argmax(-1), yy)
    best_g = max((v, k) for k, v in global_curve["train"].items())
    report["steer_global_alpha_test"] = global_curve["test"][best_g[1]]
    report["steer_global_best_alpha"] = float(best_g[1])

    def fusion(betas, temps):
        mu = class_means(h_alp_tr, y_tr, K)
        dv_tr = (torch.cdist(h_alp_tr, mu)) ** 2
        dv_te = (torch.cdist(h_alp_te, mu)) ** 2
        results = {}
        for temp in temps:
            p_tr = torch.softmax(-dv_tr / temp, -1)
            p_te = torch.softmax(-dv_te / temp, -1)
            for beta in betas:
                z_tr = logits_tr + beta * torch.log(p_tr + 1e-9)
                z_te = logits_te + beta * torch.log(p_te + 1e-9)
                results.setdefault("train", {})[(temp, beta)] = acc(z_tr.argmax(-1), y_tr)
                results.setdefault("test", {})[(temp, beta)] = acc(z_te.argmax(-1), y_te)
        return results, p_te

    med = float(((torch.cdist(h_alp_tr, class_means(h_alp_tr, y_tr, K))) ** 2).median())
    temps = [med * t for t in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)]
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    fus, p_te = fusion(betas, temps)
    best_f = max((v, k) for k, v in fus["train"].items())
    report["fusion_best_temp_beta"] = [float(best_f[1][0]), float(best_f[1][1])]
    report["fusion_test"] = fus["test"][best_f[1]]
    report["fusion_train"] = fus["train"][best_f[1]]
    report["fusion_curves"] = {k: {f"{kk[0]:.0f}/{kk[1]}": round(vv, 4) for kk, vv in v.items()} for k, v in fus.items()}
    report["alpaca_probe_report_mass"] = float(p_te.max(-1).values.mean())

    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if "curves" not in k}, indent=2))


if __name__ == "__main__":
    main()
