# ANIP Challenge - Reconnaissance Faciale, Estimation Âge & OCR Fraude

## Organisation du projet
Ce projet suit un pattern **MVC** :
- **src/models/** : définitions des modèles
- **src/controllers/** : pipelines d’entraînement & inférence
- **src/views/** : visualisations & métriques
- **src/data/** : dataloaders
- **src/utils/** : configs & fonctions utilitaires
- **models/** : poids sauvegardés
- **results/** : prédictions CSV/JSON
- **notebooks/** : démonstrations par tâche

## Environnements
- Python 3.9
- PyTorch 2.8.0
- TensorFlow 2.20.0
- Transformers 4.56.2
- OpenCV 4.11.0
- Tesseract OCR 5.5.1

## Tâches
- **Tâche 1** : Reconnaissance faciale
- **Tâche 2** : Estimation de l’âge
- **Tâche 3** : OCR & Détection de fraude
