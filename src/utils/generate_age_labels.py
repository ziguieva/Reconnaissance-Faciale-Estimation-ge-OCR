import os, csv, re, random

def generate_age_csv(img_dir: str, output_csv: str):
    """
    Parcourt img_dir et extrait l'âge depuis le pattern de nom: XXXXXX_YZWW.ext
      - XXXXXX : id personne
      - Y      : index photo
      - Z      : sexe (M/F)
      - WW     : âge (2 chiffres)
    Exemples valides: 012345_0M32.jpg, 000001_1F45.PNG
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # accepte .jpg/.jpeg/.png (casse insensible)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    rows = []

    for fname in os.listdir(img_dir):
        if not fname.endswith(exts):
            continue

        # split sur "_" puis extraire âge dans la 2e partie (ex: "0M32")
        parts = fname.split("_")
        if len(parts) != 2:
            # nom inattendu → on ignore
            continue
        right = parts[1]  # "0M32.jpg"
        right = os.path.splitext(right)[0]  # "0M32"

        # on attend: index(1 chiffre ou +) + sexe(M/F) + age(2 chiffres)
        # exemple: "0M32", "12F27"
        m = re.match(r"^\d+[MF](\d{2})$", right)
        if not m:
            continue

        age = int(m.group(1))
        rows.append((fname, age))

    if not rows:
        raise RuntimeError(f"Aucune image valide trouvée dans {img_dir}")

    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "age"])
        w.writerows(rows)

    print(f"✅ {len(rows)} lignes écrites dans {output_csv}")

def split_train_val(csv_path: str, out_train_csv: str, out_val_csv: str, val_ratio: float = 0.1, seed: int = 42):
    import pandas as pd
    df = pd.read_csv(csv_path)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)

    cut = int(len(idx) * (1 - val_ratio))
    train_idx, val_idx = idx[:cut], idx[cut:]

    df.iloc[train_idx].to_csv(out_train_csv, index=False)
    df.iloc[val_idx].to_csv(out_val_csv, index=False)

    print(f"✅ Split: {len(train_idx)} train / {len(val_idx)} val")

if __name__ == "__main__":
    # à lancer depuis la racine du projet: ANIP/
    base = "data/tache2"
    all_csv = os.path.join(base, "labels_all.csv")
    train_csv = os.path.join(base, "train_labels.csv")
    val_csv   = os.path.join(base, "val_labels.csv")

    generate_age_csv(os.path.join(base, "train"), all_csv)
    split_train_val(all_csv, train_csv, val_csv, val_ratio=0.1, seed=42)
