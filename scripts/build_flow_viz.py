"""Build the circuit flow visualization by injecting data into the HTML template."""
import json
import numpy as np
from pathlib import Path

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
LANGS = ["en", "es", "fr", "ru", "tr", "sw", "qu"]
N_LAYERS = 18
N_HEADS = 8


def load_lang_data(lang):
    """Load all available data for one language."""
    out = {}

    # Patching (head importance)
    p = RESULTS / f"patching_{lang}.npz"
    if p.exists():
        d = np.load(p)
        out["patching"] = d["head_out"].tolist()
        out["mlp_patching"] = d["mlp_out"].tolist() if "mlp_out" in d else [0] * N_LAYERS

    # EAP
    p = RESULTS / f"edge_patching_{lang}.npz"
    if p.exists():
        d = np.load(p)
        labels = [str(x) for x in d["component_labels"]]
        scores = d["node_scores"]
        eap_heads = [[0.0] * N_HEADS for _ in range(N_LAYERS)]
        eap_mlp = [0.0] * N_LAYERS
        for lbl, score in zip(labels, scores):
            if lbl.startswith("MLP"):
                layer = int(lbl[3:])
                eap_mlp[layer] = float(score)
            else:
                import re
                m = re.match(r"L(\d+)H(\d+)", lbl)
                if m:
                    eap_heads[int(m[1])][int(m[2])] = float(score)
        out["eap_heads"] = eap_heads
        out["eap_mlp"] = eap_mlp

    # Weight importance (circuit map)
    p = RESULTS / f"circuit_map_{lang}.npz"
    if p.exists():
        d = np.load(p)
        out["head_importance"] = d["head_importance"].tolist()
        out["mlp_importance"] = d["mlp_importance"].tolist()

        # Connection matrix → edge list (only forward, above threshold)
        conn = d["connection_matrix"]
        conn_labels = [str(x) for x in d["connection_labels"]]
        edges = []
        for i in range(len(conn_labels)):
            for j in range(len(conn_labels)):
                w = float(conn[i, j])
                if abs(w) < 0.05:
                    continue
                # Parse layers
                src_lbl = conn_labels[i]
                dst_lbl = conn_labels[j]
                import re
                ms = re.match(r"L(\d+)H(\d+)", src_lbl)
                md = re.match(r"L(\d+)H(\d+)", dst_lbl)
                if ms and md:
                    sl, dl = int(ms[1]), int(md[1])
                    if sl < dl:  # forward only
                        edges.append({
                            "src": src_lbl,
                            "dst": dst_lbl,
                            "weight": round(w, 4),
                            "srcLayer": sl,
                            "dstLayer": dl,
                        })

        # Keep top 200 edges per language to avoid overwhelming the viz
        edges.sort(key=lambda e: abs(e["weight"]), reverse=True)
        edges = edges[:200]
        out["connections"] = edges

    return out


def main():
    data = {}
    for lang in LANGS:
        print(f"Loading {lang}...")
        data[lang] = load_lang_data(lang)
        n_edges = len(data[lang].get("connections", []))
        print(f"  {n_edges} edges")

    # Read template
    template_path = FIGURES / "viz_circuit_flow.html"
    html = template_path.read_text()

    # Inject data
    data_json = json.dumps(data, separators=(",", ":"))
    html = html.replace("__DATA_PLACEHOLDER__", data_json)

    # Write final
    out_path = FIGURES / "viz_circuit_flow.html"
    out_path.write_text(html)
    print(f"\nWritten to {out_path}")
    print(f"Data size: {len(data_json) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
