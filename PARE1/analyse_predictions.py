"""Analyse descriptive hors-ligne de prédictions sur images civiles.

Entrée : CSV exporté après évaluation. Ne charge aucun modèle et ne produit
aucune attaque, prédiction ou recommandation opérationnelle.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REQUIRED = ("source_id", "sample_id", "split", "y_true", "resnet_pred", "deit_pred")


def read_predictions(path: Path, split: str = "test") -> list[dict[str, str]]:
    """Lit les prédictions et refuse la fuite de sources entre partitions."""
    selected = []
    samples_seen: set[str] = set()
    source_split: dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Fichier CSV vide ou sans en-tête.")
        missing = set(REQUIRED) - set(reader.fieldnames)
        if missing:
            raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))

        for number, record in enumerate(reader, start=2):
            row = {field: (record.get(field) or "").strip() for field in REQUIRED}
            if any(not value for value in row.values()):
                raise ValueError(f"Ligne {number}: champ obligatoire vide.")
            sample = row["sample_id"]
            source = row["source_id"]
            partition = row["split"].lower()
            if sample in samples_seen:
                raise ValueError(f"sample_id dupliqué : {sample}")
            samples_seen.add(sample)
            if source in source_split and source_split[source] != partition:
                raise ValueError(
                    f"Fuite entre partitions : source_id={source} "
                    f"({source_split[source]} et {partition})"
                )
            source_split[source] = partition
            if partition == split.lower():
                selected.append(row)

    if not selected:
        raise ValueError(f"Aucune ligne dans la partition '{split}'.")
    return selected


def score(rows: list[dict[str, str]], column: str) -> dict:
    """Accuracy, rappel macro et matrice de confusion, sans pondération cachée."""
    if not rows:
        raise ValueError("Aucune prédiction à analyser.")
    true_classes = sorted({row["y_true"] for row in rows})
    predicted_classes = {row[column] for row in rows}
    labels = sorted(set(true_classes) | predicted_classes)
    confusion = {
        true: {pred: 0 for pred in labels}
        for true in labels
    }
    for row in rows:
        confusion[row["y_true"]][row[column]] += 1
    recalls = {}
    for true in true_classes:
        denom = sum(confusion[true].values())
        recalls[true] = confusion[true][true] / denom
    return {
        "n": len(rows),
        "accuracy": sum(r["y_true"] == r[column] for r in rows) / len(rows),
        "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "class_support": {
            cls: sum(r["y_true"] == cls for r in rows)
            for cls in true_classes
        },
        "recall_per_class": recalls,
        "confusion_matrix": confusion,
    }


def percentile(values: list[float], q: float) -> float:
    """Quantile interpolé à la position (n-1)*q, avec 0 <= q <= 1."""
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile non défini.")
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    before = int(at)
    after = min(before + 1, len(ordered) - 1)
    return ordered[before] + (at - before) * (ordered[after] - ordered[before])


def compare(rows: list[dict[str, str]], repetitions: int = 2000, seed: int = 42) -> dict:
    """Comparaison appariée; bootstrap par image source, pas par recadrage."""
    if repetitions < 100:
        raise ValueError("Utiliser au moins 100 rééchantillonnages.")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    sources = sorted(by_source)
    a = score(rows, "resnet_pred")
    b = score(rows, "deit_pred")
    observed = b["accuracy"] - a["accuracy"]
    delta_samples = []
    rng = random.Random(seed)
    if len(sources) >= 2:
        for _ in range(repetitions):
            chosen = rng.choices(sources, k=len(sources))
            sampled = [row for source in chosen for row in by_source[source]]
            delta_samples.append(
                score(sampled, "deit_pred")["accuracy"]
                - score(sampled, "resnet_pred")["accuracy"]
            )
    return {
        "n_source_images": len(sources),
        "n_crops": len(rows),
        "resnet50": a,
        "deit_tiny": b,
        "paired_accuracy_gap_deit_minus_resnet": observed,
        "cluster_bootstrap_95pct_ci": (
            [percentile(delta_samples, .025), percentile(delta_samples, .975)]
            if delta_samples else None
        ),
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions if delta_samples else 0,
        "statistical_note": (
            "Intervalle descriptif bootstrap par image source. "
            "Ce calcul n'établit ni causalité, ni robustesse sous attaque, "
            "ni validité militaire ou embarquée."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("PARE1/resultats_descriptifs.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    rows = read_predictions(args.csv, split=args.split)
    report = compare(rows, repetitions=args.bootstrap, seed=args.seed)
    report["selected_split"] = args.split
    report["input_file"] = str(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rapport écrit : {args.out}")
    print(f"Images sources : {report['n_source_images']} ; recadrages : {report['n_crops']}")
    print(f"Accuracy ResNet-50 : {report['resnet50']['accuracy']:.3f}")
    print(f"Accuracy DeiT-Tiny : {report['deit_tiny']['accuracy']:.3f}")
    print(f"Écart apparié : {report['paired_accuracy_gap_deit_minus_resnet']:+.3f}")
    print(f"IC descriptif bootstrap : {report['cluster_bootstrap_95pct_ci']}")


if __name__ == "__main__":
    main()
