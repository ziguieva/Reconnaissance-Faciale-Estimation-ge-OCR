import os
import random
import csv

def generate_pairs(data_dir, output_file, num_negatives=1):
    """
    Génère un fichier CSV de paires (positives et négatives).
    - data_dir : dossier contenant les images (ex: train/)
    - output_file : chemin du CSV à créer
    - num_negatives : combien de paires négatives par paire positive
    """
    persons = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            person_id = fname.split("_")[0]  # ex: "0001" si fichier = "0001_0.jpg"
            persons.setdefault(person_id, []).append(fname)

    pairs = []

    # Paires positives (2 photos de la même personne)
    for pid, files in persons.items():
        if len(files) >= 2:
            pairs.append((files[0], files[1], 1))

    # Paires négatives (personnes différentes)
    all_ids = list(persons.keys())
    for pid, files in persons.items():
        other_ids = [x for x in all_ids if x != pid]
        for f in files[:1]:  # prend 1 photo par personne pour générer du négatif
            for _ in range(num_negatives):
                neg_pid = random.choice(other_ids)
                neg_file = random.choice(persons[neg_pid])
                pairs.append((f, neg_file, 0))

    # Sauvegarde CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for img1, img2, label in pairs:
            writer.writerow([img1, img2, label])

    print(f"✅ Fichier {output_file} généré avec {len(pairs)} paires.")

if __name__ == "__main__":
    generate_pairs("data/tache1/train", "data/tache1/train_pairs.csv")
    generate_pairs("data/tache1/test", "data/tache1/test_pairs.csv")
