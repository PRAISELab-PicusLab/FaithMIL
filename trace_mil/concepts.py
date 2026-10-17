"""Histological concepts of the report: QuiltNet-B-16 zero-shot matching, linked to ontology terms.

Concepts (14 tissue concepts and an artifact class) are linked to terms of the NCI Thesaurus and of UBERON. The text
embedding of a concept is the normalised mean of (i) the embeddings of its hand-written short descriptions and (ii) the
embeddings of the labels and synonyms of its ontology terms (for the three Gleason patterns also the morphological
descriptors listed in the NCIt definitions; the artifact class uses technical prompts only, no clinical ontology covers it).

    score of a patch   z_k = (cos_k - mu_k) / sd_k, with mu_k, sd_k estimated on random patches without using the outcome,
                       then centred across the concepts of the patch (the raw z_k share a large common component)
    score of a slide   mean of the centred patch scores
    evidence           a concept is shown for a patient only if the C-index of its slide score, in the patient's cohort,
                       is >= 0.55 or <= 0.45 (cohort-level evidence); otherwise the report abstains
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .layout import Layout
from .utils import c_index

QUILT = "hf-hub:wisdomik/QuiltNet-B-16"
PREFIX = "a histopathology image of "
MAX_ROWS = 3000  # slide scores use at most this many random patches per slide
EVIDENCE = (0.45, 0.55)
# coarse tissue class of the concepts, to name the cohort patterns
GLANDULAR = {"benign", "atrophy", "gleason3", "gleason4", "cribriform", "gleason5", "intraductal", "necrosis"}
STROMAL = {"stroma", "inflammation", "vessels", "perineural"}

CONCEPTS = {
    "benign": ("benign glands", [
        "benign prostatic glands with two cell layers", "normal prostate tissue with benign glands",
        "benign prostate glands with papillary infoldings"]),
    "atrophy": ("atrophic glands", [
        "atrophic prostate glands with scant cytoplasm", "prostatic atrophy", "post-atrophic hyperplasia of the prostate"]),
    "gleason3": ("well-formed tumour glands (Gleason 3)", [
        "Gleason pattern 3 prostate adenocarcinoma", "small well-formed separate neoplastic glands",
        "low grade prostate cancer with discrete glands"]),
    "gleason4": ("fused / poorly formed glands (Gleason 4)", [
        "Gleason pattern 4 prostate adenocarcinoma", "fused and poorly formed glands in prostate cancer",
        "glomeruloid glands in prostate adenocarcinoma"]),
    "cribriform": ("cribriform glands", [
        "cribriform prostate cancer", "cribriform glands with punched-out lumina",
        "large cribriform gland in prostate adenocarcinoma"]),
    "gleason5": ("solid sheets / single tumour cells (Gleason 5)", [
        "Gleason pattern 5 prostate adenocarcinoma", "solid sheets of tumor cells without gland formation",
        "single infiltrating tumor cells in prostate cancer"]),
    "necrosis": ("comedonecrosis / necrosis", [
        "comedonecrosis in prostate cancer", "tumor necrosis", "central necrosis in a cribriform gland"]),
    "intraductal": ("intraductal carcinoma", [
        "intraductal carcinoma of the prostate", "malignant cells filling and expanding prostatic ducts",
        "intraductal spread of prostate cancer"]),
    "perineural": ("nerves / tumour around nerves", [
        "perineural invasion by prostate cancer", "tumor glands surrounding a nerve", "nerve bundle in prostate"]),
    "stroma": ("fibromuscular stroma", [
        "prostatic fibromuscular stroma", "smooth muscle stroma without glands", "dense collagenous stroma"]),
    "inflammation": ("inflammation", [
        "chronic inflammation with lymphocytes in prostate", "dense inflammatory infiltrate", "lymphocytic inflammation"]),
    "fat": ("fat (possible extraprostatic tissue)", [
        "adipose tissue", "periprostatic fat", "tumor glands invading fat"]),
    "seminal_vesicle": ("seminal vesicle", [
        "seminal vesicle epithelium with lipofuscin", "seminal vesicle tissue", "seminal vesicle invasion"]),
    "vessels": ("blood vessels / haemorrhage", [
        "blood vessels in stroma", "hemorrhage with red blood cells", "thick-walled blood vessels"]),
    "artifact": ("artefact (fold, blur, pen, edge)", [
        "tissue fold artifact", "blurry out of focus tissue", "pen marker ink on a histology slide"]),
}


MAPPING = {
    "benign": [("UBERON", "prostate glandular acinus"), ("NCIT", "C13103")],
    "atrophy": [("NCIT", "C97114")],
    "gleason3": [("NCIT", "C28087"), ("NCIT", "C162656")],
    "gleason4": [("NCIT", "C28088"), ("NCIT", "C162657")],
    "cribriform": [("NCIT", "C39895"), ("NCIT", "C162973")],
    "gleason5": [("NCIT", "C28089"), ("NCIT", "C162658")],
    "necrosis": [("NCIT", "C36128"), ("NCIT", "C35957")],
    "intraductal": [("NCIT", "C161022")],
    "perineural": [("NCIT", "C48260")],
    "stroma": [("UBERON", "prostate gland stroma"), ("NCIT", "C128569")],
    "inflammation": [("NCIT", "C174450"), ("NCIT", "C26866")],
    "fat": [("UBERON", "adipose tissue"), ("NCIT", "C12472")],
    "seminal_vesicle": [("UBERON", "seminal vesicle"), ("NCIT", "C12787")],
    "vessels": [("UBERON", "blood vessel"), ("NCIT", "C12679"), ("NCIT", "C26791")],
    "artifact": [("NCIT", "C75603")],
}


def clean_synonyms(label, syns):
    """Synonyms usable as a short prompt: no abbreviations / codes / 'Grade Group' / bare 'Pattern n'; label first; <= 4."""
    out = []
    for x in [label] + list(syns):
        x = x.strip()
        bad = (len(x) <= 4 or x.isupper() or "(" in x or re.match(r"^(grade group|pattern \d|present$|absent$)", x.lower())
               or ("," in x and x.split(",")[0].isupper()))
        if x and not bad and x.lower() not in [o.lower() for o in out]:
            out.append(x)
    return out[:4] or [label]


def short(text, n=35):
    text = re.sub(r"\s*\((ISUP|WHO|NCI|ACC)[^)]*\)\.?\s*$", "", text.strip())
    w = text.split()
    return " ".join(w[:n]).rstrip(" ;,") + ("" if len(w) <= n else " ...")


def load_ncit(zip_path, version):
    with zipfile.ZipFile(zip_path) as z:
        rows = [l.split("\t") for l in z.read("Thesaurus.txt").decode("utf8").splitlines()]
    db = {}
    for r in rows:
        syn = r[3].split("|")
        db[r[0]] = {"id": "NCIt:" + r[0], "label": syn[0], "definition": r[4], "synonyms": clean_synonyms(syn[0], syn[1:])}
    return db, "NCI Thesaurus " + version


def load_uberon(obo_path):
    db, cur, ver = {}, None, ""
    for l in open(obo_path, encoding="utf8"):
        l = l.rstrip("\n")
        if l.startswith("data-version:"):
            ver = l.split(": ", 1)[1]
        if l == "[Term]":
            cur = {"definition": "", "syn": []}
        elif l.startswith("["):
            cur = None
        elif cur is not None:
            if l.startswith("id: "):
                cur["id"] = l[4:]
            elif l.startswith("name: "):
                cur["label"] = l[6:]
                db[cur["label"]] = cur
            elif l.startswith("def: ") and '"' in l:
                cur["definition"] = l.split('"')[1]
            elif l.startswith("synonym: ") and "EXACT" in l and '"' in l:
                cur["syn"].append(l.split('"')[1])
    for t in db.values():
        t["synonyms"] = clean_synonyms(t["label"], t.pop("syn"))
    return db, "UBERON " + ver


# ----------------------------------------------------------------------------- ontology terms
def build_terms(ncit_zip: Path, uberon_obo: Path, ncit_version: str, out: Path | None = None) -> dict:
    """concept -> {"terms": [{id, label, synonyms, definition, definition_full, version}]} from the ontology files."""
    ncit, v1 = load_ncit(ncit_zip, ncit_version)
    ub, v2 = load_uberon(uberon_obo)
    res, table = {}, []
    for concept, specs in MAPPING.items():
        items = []
        for onto, key in specs:
            t = (ncit if onto == "NCIT" else ub)[key]
            items.append({"id": t["id"], "label": t["label"], "synonyms": t["synonyms"], "definition": short(t["definition"]),
                          "definition_full": re.sub(r"\s*\((ISUP|WHO|NCI|ACC)[^)]*\)\.?\s*$", "", t["definition"].strip()),
                          "version": v1 if onto == "NCIT" else v2})
            table.append([concept, t["id"], t["label"], " | ".join(t["synonyms"]), short(t["definition"])])
        res[concept] = {"terms": items}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "terms.json").write_text(json.dumps(res, indent=1))
        with open(out / "terms_table.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concept", "id", "label", "synonyms", "definition_used"])
            w.writerows(table)
    return res


# ----------------------------------------------------------------------------- text embeddings
norm = lambda x: torch.nn.functional.normalize(x, dim=-1)  # noqa: E731


def clause_prompts(concept: dict) -> list:
    """Morphological clauses of the ontology definitions that are lists of descriptors separated by ';' (ISUP 2014 grade
    patterns in NCIt); encyclopedic definitions ('a finding that indicates ...') are not used."""
    out = []
    for x in concept["terms"]:
        d = x["definition_full"]
        if d.count(";") >= 1 and not re.match(r"^(An?|The) ", d):
            out += [c.strip(" .") for c in d.split(";") if 2 <= len(c.split()) <= 20]
    return out


def concept_embeddings(terms: dict, device="cpu"):
    """(names, descriptions, embeddings (K, 512)) of the concepts."""
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(QUILT)
    tok = open_clip.get_tokenizer(QUILT)
    model = model.to(device).eval()

    def enc(texts):
        with torch.no_grad():
            return norm(norm(model.encode_text(tok(texts).to(device)).float()).mean(0, keepdim=True))[0].cpu()

    names, descs, embs = list(CONCEPTS), [], []
    for n in names:
        desc, prompts = CONCEPTS[n]
        manual = enc([PREFIX + p for p in prompts] + prompts)
        if n == "artifact":
            hyb = manual
        else:
            syn = [s for x in terms[n]["terms"] for s in x["synonyms"]]
            extra = syn + clause_prompts(terms[n])
            hyb = norm(manual + enc([PREFIX + s for s in extra] + extra))
        descs.append(desc if n == "artifact" else f"{desc} [{terms[n]['terms'][0]['id']}]")
        embs.append(hyb)
    return names, descs, torch.stack(embs)


# ----------------------------------------------------------------------------- scores
def centred(z: np.ndarray) -> np.ndarray:
    return z - z.mean(axis=-1, keepdims=True)


def build(layout: Layout, terms: dict, device="cpu", seed: int = 0) -> dict:
    """Concept embeddings, normalisation statistics and cohort-level validation; written to report/."""
    names, descs, T = concept_embeddings(terms, device)
    df = pd.read_csv(layout.folds_csv)
    cases = df.case_id.tolist()
    ss = np.random.SeedSequence(seed).spawn(len(cases))
    feat_dir = layout.features("quiltnet", "children")
    slide_cos, sample = {}, []
    for c, s in zip(cases, ss):
        rng = np.random.default_rng(s)
        A = np.load(feat_dir / f"{c}.npy", mmap_mode="r")
        rows = np.sort(rng.choice(len(A), size=MAX_ROWS, replace=False)) if len(A) > MAX_ROWS else slice(None)
        cos = (norm(torch.from_numpy(np.array(A[rows])).float()) @ T.T).numpy()
        slide_cos[c] = cos.mean(0)
        sample.append(cos[rng.choice(len(cos), size=min(500, len(cos)), replace=False)])
    S = np.concatenate(sample)
    mu, sd = S.mean(0), S.std(0)
    z = pd.DataFrame({c: centred((v - mu) / sd) for c, v in slide_cos.items()}, index=names).T
    z.index.name = "case_id"
    m = df.set_index("case_id").join(z)
    rows = []
    for n, d in zip(names, descs):
        r = {"concept": n, "description": d}
        for coh, g in m.groupby("cohort"):
            r[f"C_{coh}"] = c_index(g.follow_up_years.to_numpy(), g[n].to_numpy(), g.event.to_numpy())
        rows.append(r)
    val = pd.DataFrame(rows)
    out = layout.report
    out.mkdir(parents=True, exist_ok=True)
    z.to_csv(out / "slide_concepts.csv")
    val.to_csv(out / "concept_validation.csv", index=False)
    pack = {"names": names, "descriptions": descs, "text": T, "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd),
            "terms": terms, "model": QUILT}
    torch.save(pack, out / "concepts.pt")
    return pack


def load(layout: Layout) -> tuple[dict, pd.DataFrame]:
    pack = torch.load(layout.report / "concepts.pt", weights_only=False)
    return pack, pd.read_csv(layout.report / "concept_validation.csv").set_index("concept")


def patch_z(pack: dict, Q: np.ndarray) -> np.ndarray:
    """Centred relative scores (n, K) of QuiltNet image embeddings Q (n, 512)."""
    cos = (norm(torch.from_numpy(np.asarray(Q, np.float32))) @ pack["text"].T).numpy()
    return centred((cos - pack["mu"].numpy()) / pack["sd"].numpy())


def usable(validation: pd.DataFrame, cohort: str) -> list:
    """Concepts with cohort-level evidence in `cohort` (the artifact class is never a tissue concept)."""
    c = validation[f"C_{cohort}"]
    return [n for n in validation.index if n != "artifact" and (c[n] >= EVIDENCE[1] or c[n] <= EVIDENCE[0])]
