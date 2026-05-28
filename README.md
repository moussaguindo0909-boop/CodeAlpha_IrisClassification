# Classification des fleurs Iris

**Stage Data Science CodeAlpha — Tâche 1**
Stagiaire : Moussa Guindo | ID : CA/DF1/100834 | Juin 2026

---

## Présentation

Ce projet entraîne et compare plusieurs modèles de machine learning pour classifier automatiquement les espèces de fleurs Iris à partir de leurs mesures (longueur et largeur des sépales et pétales). Le modèle le plus performant est sélectionné automatiquement et sauvegardé pour une utilisation ultérieure.

**Espèces cibles :** Iris Setosa, Iris Versicolor, Iris Virginica

---

## Structure du projet

```
CodeAlpha_IrisClassification/
│
├── iris_classification.py      # Script principal
├── iris_best_model.pkl         # Meilleur modèle sauvegardé
├── iris_scaler.pkl             # StandardScaler sauvegardé
├── iris_exploration.png        # Graphiques d'exploration des données
├── iris_confusion_matrix.png   # Matrice de confusion du meilleur modèle
├── iris_model_comparison.png   # Comparaison des performances
└── README.md
```

---

## Technologies

| Outil | Usage |
|---|---|
| Python 3.x | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Modèles, prétraitement, évaluation |
| Matplotlib / Seaborn | Visualisation |
| Joblib | Sauvegarde du modèle |

---

## Modèles entraînés

| Modèle | Configuration |
|---|---|
| K-Nearest Neighbors | k = 5 |
| Random Forest | 100 estimateurs, random_state = 42 |
| Support Vector Machine | Noyau RBF, C = 1.0 |

---

## Résultats

| Modèle | Accuracy |
|---|---|
| KNN (k=5) | 96.67% |
| Random Forest | 100.00% |
| SVM (RBF) | 100.00% |

Le meilleur modèle est sélectionné et sauvegardé automatiquement sous `iris_best_model.pkl`.

---

## Lancer le projet

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/CodeAlpha_IrisClassification.git
cd CodeAlpha_IrisClassification

# Installer les dépendances
pip install numpy pandas matplotlib seaborn scikit-learn joblib

# Exécuter le script
python iris_classification.py
```

---

## Visualisations

### Exploration des données
![Exploration](iris_exploration.png)

### Matrice de confusion
![Matrice de confusion](iris_confusion_matrix.png)

### Comparaison des modèles
![Comparaison](iris_model_comparison.png)

---

## Auteur

**Moussa Guindo**
Stagiaire Data Science — CodeAlpha
ID : CA/DF1/100834