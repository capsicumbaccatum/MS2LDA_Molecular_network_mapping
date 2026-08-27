# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:48:13 2026

@author: felic
"""

#!/usr/bin/env python3
"""
MS2LDA motif exporter CLI

Example:
python ms2lda_motif_exporter.py \
    --mgf spectra.mgf \
    --model ms2lda.bin \
    --viz ms2lda_viz.json \
    --outdir results/
"""

import argparse
import os
import sys
import json
import traceback

import tomotopy as tp
from matchms.importing import load_from_mgf
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Utility logging
# ---------------------------------------------------------
def log(msg):
    print(f"[INFO] {msg}")

def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    log(f"{name} OK -> {path}")


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
def load_spectra(mgf_path):
    log("Loading MGF spectra...")
    spectra = list(load_from_mgf(mgf_path))
    log(f"Loaded {len(spectra)} spectra")
    if len(spectra) == 0:
        raise RuntimeError("MGF file contains zero spectra.")
    return spectra


def load_model(model_path):
    log("Loading MS2LDA model...")
    model = tp.LDAModel.load(model_path)
    log(f"Model loaded with {len(model.docs)} documents")
    if len(model.docs) == 0:
        raise RuntimeError("Model contains zero documents.")
    return model


def load_viz_json(viz_path):
    log("Loading ms2lda_viz JSON...")

    # ---- sanity check ----
    if os.path.isdir(viz_path):
        log("Viz path is a directory → searching for ms2lda_viz.json inside")
        candidate = os.path.join(viz_path, "ms2lda_viz.json")
        if not os.path.exists(candidate):
            raise RuntimeError(
                "Provided --viz path is a directory but does not contain ms2lda_viz.json"
            )
        viz_path = candidate
        log(f"Found JSON file inside folder → {viz_path}")

    with open(viz_path, encoding="utf-8") as f:
        data = json.load(f)

    if "spectra_data" not in data:
        raise RuntimeError("Invalid viz JSON: missing 'spectra_data' key")

    log(f"Loaded metadata for {len(data['spectra_data'])} spectra")
    return data

# ---------------------------------------------------------
# Core motif extraction
# ---------------------------------------------------------
def extract_top_motifs(model, viz_json, threshold):
    log(f"Extracting motifs with probability > {threshold}")

    if len(model.docs) != len(viz_json["spectra_data"]):
        raise RuntimeError(
            f"Mismatch: model docs ({len(model.docs)}) "
            f"!= spectra metadata ({len(viz_json['spectra_data'])})"
        )

    top_motifs = []
    scans_list = []

    for i in range(len(viz_json["spectra_data"])):
        topics = model.docs[i].get_topics()

        doc_motifs = [(motif, prob) for motif, prob in topics if prob > threshold]
        top_motifs.append(doc_motifs)

        metadata = viz_json["spectra_data"][i]["metadata"]
        scans_list.append(metadata["feature_id"])

        if i % 1000 == 0:
            log(f"Processed {i} spectra")

    log("Motif extraction complete")
    return scans_list, top_motifs


# ---------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------
def build_long_table(ids, motifs):
    log("Building LONG table")
    df = pd.DataFrame({"id": ids, "motifs": motifs})

    df_long = df.explode("motifs")
    df_long[["motif", "probability"]] = pd.DataFrame(
        df_long["motifs"].tolist(),
        index=df_long.index
    )
    df_long = df_long.drop(columns="motifs")
    df_long["motif_index"] = df_long.groupby("id").cumcount() + 1

    log(f"Long table rows: {len(df_long)}")
    return df_long


def build_wide_table(ids, motifs):
    log("Building WIDE table")

    df = pd.DataFrame({"feature_id": ids, "motifs": motifs})

    df["motifs"] = df["motifs"].apply(
        lambda lst: {f"motif_{k}": v for k, v in lst}
    )

    df_wide = pd.concat(
        [df.drop(columns="motifs"),
         df["motifs"].apply(pd.Series)],
        axis=1
    )

    motif_cols = df_wide.columns.difference(["feature_id"])
    df_wide[motif_cols] = df_wide[motif_cols].fillna(0)

    log(f"Wide table shape: {df_wide.shape}")
    return df_wide


# ---------------------------------------------------------
# Main CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MS2LDA motif exporter")

    parser.add_argument("--mgf", required=True, help="MGF file")
    parser.add_argument("--model", required=True, help="Tomotopy model (.bin)")
    parser.add_argument("--viz", required=True, help="ms2lda_viz.json")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--mode", choices=["long", "wide", "both"], default="both")

    args = parser.parse_args()

    try:
        log("==== MS2LDA MOTIF EXPORT START ====")

        os.makedirs(args.outdir, exist_ok=True)

        check_file(args.mgf, "MGF")
        check_file(args.model, "Model")
        check_file(args.viz, "Viz JSON")

        spectra = load_spectra(args.mgf)
        model = load_model(args.model)
        viz_json = load_viz_json(args.viz)

        ids, motifs = extract_top_motifs(model, viz_json, args.threshold)

        if args.mode in ["long", "both"]:
            df_long = build_long_table(ids, motifs)
            long_path = os.path.join(args.outdir, "motifs_long.csv")
            df_long.to_csv(long_path, index=False)
            log(f"Saved LONG table -> {long_path}")

        if args.mode in ["wide", "both"]:
            df_wide = build_wide_table(ids, motifs)
            wide_path = os.path.join(args.outdir, "motifs_wide.csv")
            df_wide.to_csv(wide_path, index=False)
            log(f"Saved WIDE table -> {wide_path}")

        log("==== PIPELINE COMPLETE SUCCESSFULLY ====")

    except Exception as e:
        print("\n[ERROR] Pipeline failed")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
