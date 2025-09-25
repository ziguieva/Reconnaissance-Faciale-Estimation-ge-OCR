````markdown
# Installation et téléchargement de données Kaggle

Ce guide explique comment installer l’outil **Kaggle CLI**, configurer votre compte, et télécharger les données d’une compétition Kaggle.

---

## 1. Installer Kaggle CLI
Assurez-vous d’avoir **Python** et **pip** installés sur votre machine.  
Puis exécutez la commande suivante :

```bash
pip install kaggle
````


## 2. Configurer l’API Kaggle

Pour utiliser Kaggle en ligne de commande, il faut configurer une clé d’API.

1. Allez sur [Kaggle - Account](https://www.kaggle.com/account).
2. Descendez jusqu’à la section **API** et cliquez sur **Create New API Token**.
3. Un fichier `kaggle.json` sera téléchargé sur votre ordinateur.
4. Placez ce fichier dans le dossier suivant selon votre système :

   * **Linux / MacOS** : `~/.kaggle/kaggle.json`
   * **Windows** : `C:\Users\<VotreNom>\.kaggle\kaggle.json`

⚠️ Vérifiez que les permissions du fichier soient sécurisées (sur Linux/MacOS) :

```bash
chmod 600 ~/.kaggle/kaggle.json
```


## 3. S’inscrire à la compétition

Avant de télécharger les données, vous devez vous inscrire à la compétition sur le site Kaggle.
Exemple : [ANIP Reconnaissance Faciale - Estimation d’Âge - OCR](https://www.kaggle.com/competitions/anip-reconnaissance-faciale-estimation-age-ocr)

Cliquez sur **Join Competition**.

---

## 4. Télécharger les données

Utilisez la commande suivante dans votre terminal pour télécharger les fichiers de la compétition :

```bash
kaggle competitions download -c anip-reconnaissance-faciale-estimation-age-ocr
```

Les fichiers compressés seront téléchargés dans votre répertoire courant.
Vous pouvez ensuite les extraire avec :

```bash
unzip anip-reconnaissance-faciale-estimation-age-ocr.zip -d data/
```

## 5. Vérifier l’installation

Pour tester que tout fonctionne, lancez :

```bash
kaggle --version
```

Cela doit afficher la version installée de l’outil Kaggle CLI.


## ✅ Résumé

1. Installer Kaggle CLI.
2. Configurer votre clé API `kaggle.json`.
3. Vous inscrire à la compétition.
4. Télécharger les données avec `kaggle competitions download`.

```

---

Veux-tu que je t’explique aussi comment **mettre ce fichier en avant dans le `README.md`** pour que les visiteurs le voient directement ?
```
