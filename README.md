# Mini-Projet IA : Prédiction du Revenu Annuel d'un Marocain

*Auteurs du notebook (et potentiellement du projet) :* 

  -Hala QUERRICH 

  -Jihane ELAMRANI

  -Zakaria AYAD .


*Date de présentation prévue :* SAMEDI 17 Mai 2025

## 1. Introduction et Objectifs

Ce projet vise à développer un système de Machine Learning de bout en bout pour prédire le revenu annuel d'individus au Maroc. L'objectif principal est de simuler un cas d'usage réel, depuis la génération de données synthétiques jusqu'au déploiement d'un modèle prédictif via une application web.

Les objectifs spécifiques incluent :
*   *Génération de données synthétiques :* Créer un jeu de données réaliste (dataset_revenu_marocains.csv) basé sur des contraintes statistiques du Haut-Commissariat au Plan (HCP) marocain.
*   *Analyse exploratoire des données (EDA) :* Comprendre la structure, les distributions et les relations au sein des données.
*   *Prétraitement et Nettoyage des données :* Gérer les doublons, les valeurs manquantes et les valeurs aberrantes.
*   *Ingénierie des Caractéristiques (Feature Engineering) :* Transformer les données brutes en un format adapté aux modèles de Machine Learning (création/suppression de variables, normalisation, encodage).
*   *Modélisation :*
    *   Construire des pipelines de prétraitement et d'entraînement.
    *   Implémenter et comparer cinq modèles de régression : Régression Linéaire, Arbre de Décision, Forêt Aléatoire, Gradient Boosting, et Réseau de Neurones Multi-couches (MLPRegressor).
    *   Optimiser les hyperparamètres de chaque modèle en utilisant la validation croisée (GridSearchCV ou RandomizedSearchCV).
*   *Évaluation des Modèles :* Mesurer les performances des modèles à l'aide de métriques telles que le MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), et R² (Coefficient de détermination).
*   *Sélection du Meilleur Modèle :* Identifier le modèle le plus performant pour la tâche de prédiction.
*   *Déploiement :*
    *   Sauvegarder le modèle final.
    *   Créer une API RESTful avec FastAPI pour exposer les fonctionnalités de prédiction du modèle.
    *   Développer une application web interactive avec Streamlit pour permettre aux utilisateurs d'obtenir des prédictions de revenu.

## 2. Dataset : dataset_revenu_marocains.csv

Le dataset est généré par un script Python (generate_dataset.py, non fourni dans l'archive mais mentionné comme livrable) et contient environ 40 000 enregistrements.

### 2.1. Contraintes de Génération (basées sur HCP) :

*   *Revenu Annuel Moyen :* 21.949 DH/an.
    *   Milieu Urbain : 26.988 DH/an.
    *   Milieu Rural : 12.862 DH/an.
*   *Répartition des Revenus :* 71,8% des individus ont un revenu inférieur à la moyenne nationale (Urbain 65,9%, Rural 85,4%).
*   *Caractéristiques Principales et leurs relations avec le revenu :*
    1.  *Age :* Le revenu tend à augmenter avec l'âge, jusqu'à l'âge de la retraite.
    2.  *Catégorie d'âge :* Jeune, adulte, sénior, âgé.
    3.  *Sexe :* Le revenu moyen des hommes est plus élevé que celui des femmes.
    4.  *Lieu :* Urbain / Rural (les urbains gagnent tendanciellement plus).
    5.  *Niveau_Education :* Quatre niveaux (Sans niveau, Fondamental, Secondaire, Supérieur). Un niveau supérieur est associé à un revenu plus élevé.
    6.  *Experience_Annees :* Plus d'expérience est corrélé à un revenu plus élevé.
    7.  *Etat_Matrimonial :* Célibataire, marié, divorcé, veuf.
    8.  *Biens_Possedes :* (Aucun, Voiture, Maison, Terrain, combinaisons). Reflète le niveau socio-économique.
    9.  *Categorie_Socioprofessionnelle :* 6 groupes (du plus haut revenu au plus bas) :
        *   Groupe 1 : Responsables hiérarchiques fonction publique, directeurs, cadres supérieurs, professions libérales.
        *   Groupe 2 : Cadres moyens, employés, commerçants, intermédiaires commerciaux/financiers.
        *   Groupe 3 : Inactifs (retraités, rentiers, etc.).
        *   Groupe 4 : Exploitants agricoles, pêcheurs, forestiers, chasseurs, ouvriers agricoles.
        *   Groupe 5 : Conducteurs d'installations/machines, artisans, ouvriers qualifiés.
        *   Groupe 6 : Manœuvres non agricoles, manutentionnaires, petits métiers, chômeurs.
    10. *Nombre_Enfants :* Nombre d'enfants à charge.
    11. *Acces_Internet :* Oui / Non.
    12. *Type_Transport :* Voiture / Transport en commun / Marche.
    13. *Trois autres caractéristiques pertinentes* (non spécifiées dans le brief mais ajoutées par les auteurs, par ex. Redondance_Transport qui est ensuite retirée).
*   *Problèmes de données à inclure :*
    *   Valeurs manquantes.
    *   Valeurs aberrantes.
    *   Colonnes redondantes.
    *   Colonnes non pertinentes (ex: Identifiant_Aleatoire).

### 2.2. Structure du Dataset Observée dans le Notebook :
Le notebook révèle les colonnes suivantes (avant nettoyage/transformation) :
Age, Sexe, Lieu, Niveau_Education, Experience_Annees, Etat_Matrimonial, Biens_Possedes, Categorie_Socioprofessionnelle, Nombre_Enfants, Acces_Internet, Type_Transport, Revenu_Annuel (cible), Redondance_Transport, Identifiant_Aleatoire.

## 3. Structure du Projet

Le projet est constitué des fichiers suivants :
*   IA_Projet_hala_zakaria_jihane_(3) (2).ipynb: Notebook Jupyter contenant l'analyse, le prétraitement, l'entraînement des modèles et l'évaluation.
*   dataset_revenu_marocains.csv: Jeu de données.
*   rapport_AI_analyse.html: Rapport d'analyse exploratoire généré par Sweetviz.
*   modele_selection.joblib: Pipeline du modèle de Machine Learning final sauvegardé.
*   api.py: Script FastAPI pour l'API de prédiction.
*   app.py: Script Streamlit pour l'application web.
*   generate_dataset.py: Script pour générer dataset_revenu_marocains.csv.
*   README.md: Documentation.

## 4. Stack Technique

*   *Langage de Programmation :* Python (version 3.11.9 utilisée dans le notebook)
*   *Manipulation de Données :* NumPy (v1.23.5), Pandas
*   *Visualisation :* Matplotlib, Seaborn
*   *Analyse Exploratoire Automatique :* Sweetviz
*   *Apprentissage Automatique (Machine Learning) :* Scikit-learn
*   *Sauvegarde/Chargement de Modèle :* Joblib
*   *Développement d'API :* FastAPI, Uvicorn
*   *Développement d'Application Web :* Streamlit
*   *Validation de Données (API) :* Pydantic
*   *Environnement :* Jupyter Notebook / Google Collab

## 5. Méthodologie du Pipeline de Machine Learning

Le notebook détaille les étapes suivantes :

### 5.1. Importation des Librairies
Importation de tous les modules nécessaires pour chaque étape du projet.

### 5.2. Chargement et Aperçu Initial des Données
Le dataset dataset_revenu_marocains.csv est chargé dans un DataFrame Pandas. Un premier aperçu (df.head(10)) est effectué.

### 5.3. Analyse Exploratoire des Données (EDA)
*   *Résumé des Données :*
    *   len(df) : Volume total d'instances (40000 initialement).
    *   df.shape : Dimensions du dataset (40000 lignes, 14 colonnes initialement).
    *   df.dtypes : Types de données de chaque colonne (mélange d'int64, object, float64).
    *   df.describe(include='all') : Statistiques descriptives pour les colonnes numériques et catégorielles.
*   *Analyse Automatique avec Sweetviz :*
    *   Génération d'un rapport HTML (rapport_AI_analyse.html) pour visualiser les distributions, les valeurs manquantes, les corrélations, et les caractéristiques de chaque variable.

### 5.4. Nettoyage des Données
*   *Gestion des Doublons :*
    *   df.duplicated().sum() est utilisé pour vérifier les doublons.
    *   df.drop_duplicates() pour les supprimer. (Aucun doublon n'a été trouvé dans l'exécution présentée).
*   *Gestion des Valeurs Manquantes :*
    *   df.isnull().sum() pour identifier les colonnes avec des NaN.
    *   Les colonnes catégorielles Niveau_Education, Etat_Matrimonial, Biens_Possedes (chacune avec 2000 NaN) sont imputées avec leur *mode* respectif (valeur la plus fréquente).
*   *Gestion des Valeurs Aberrantes (Outliers) :*
    *   Uniquement sur la variable cible Revenu_Annuel.
    *   Calcul du 99ème percentile : revenu_99 = df['Revenu_Annuel'].quantile(0.99).
    *   Filtrage : df = df[df['Revenu_Annuel'] < revenu_99]. 400 outliers ont été supprimés, réduisant le dataset à 39600 instances.

### 5.5. Préparation des Données et Pipeline de Prétraitement
*   *Sélection des Caractéristiques (X) et de la Cible (y) :*
    *   y = df['Revenu_Annuel']
    *   X = df.drop(['Revenu_Annuel', 'Identifiant_Aleatoire', 'Redondance_Transport'], axis=1)
        *   Identifiant_Aleatoire : Non pertinent.
        *   Redondance_Transport : Supposée redondante ou peu informative (décision à justifier plus en détail si possible).
*   *Division en Ensembles d'Entraînement et de Test :*
    *   train_test_split(X, y, test_size=0.3, random_state=42)
    *   Taille entraînement : 27720, Taille test : 11880.
*   **Construction du Préprocesseur (ColumnTransformer) :**
    *   Identification des colonnes numériques et catégorielles : X.select_dtypes().
    *   **Pipeline Numérique (num_pipeline) :**
        1.  SimpleImputer(strategy='median') : Imputation des NaN par la médiane.
        2.  StandardScaler() : Normalisation (centrage-réduction).
    *   **Pipeline Catégoriel (cat_pipeline) :**
        1.  SimpleImputer(strategy='most_frequent') : Imputation des NaN par le mode.
        2.  OneHotEncoder(handle_unknown='ignore') : Encodage One-Hot, les catégories inconnues lors du test seront ignorées (toutes les colonnes à 0).
    *   preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])
    *   Application : X_train_preprocessed = preprocessor.fit_transform(X_train) et X_test_preprocessed = preprocessor.transform(X_test).

### 5.6. Construction, Entraînement et Validation des Modèles
*   **Définition des Modèles et Espaces d'Hyperparamètres (model_config) :**
    1.  *LinearRegression :* Aucun hyperparamètre spécifique à tuner via GridSearchCV.
    2.  *DecisionTreeRegressor :*
        *   criterion: ['squared_error', 'absolute_error']
        *   max_depth: [None, 5, 6, 7, 10]
        *   min_samples_split: [2, 3, 4, 5, 10]
    3.  *RandomForestRegressor :* (Configuration réduite pour un entraînement plus rapide)
        *   n_estimators: [50, 100, 150]
        *   criterion: ['squared_error']
        *   max_depth: [None, 10, 20]
    4.  *GradientBoostingRegressor :*
        *   loss: ['squared_error', 'absolute_error']
        *   learning_rate: [0.01, 0.1, 0.2]
        *   n_estimators: [100, 200, 300]
        *   subsample: [0.5, 0.8, 1.0]
    5.  *MLPRegressor :* (Réseau de Neurones Multi-couches, early_stopping=True)
        *   hidden_layer_sizes: [(50,), (100,), (100, 50)]
        *   activation: ['relu', 'tanh', 'logistic']
        *   solver: ['adam'] 
        *   alpha: [0.0001, 0.001, 0.01]
        *   learning_rate: ['constant', 'adaptive']
        *   learning_rate_init: [0.001, 0.01, 0.1]
        *   max_iter: [200, 300]
*   *Processus d'Entraînement et d'Évaluation :*
    *   Une fonction train_and_evaluate_models est définie pour itérer sur model_config.
    *   Pour chaque modèle, un Pipeline scikit-learn est créé, intégrant le preprocessor et le modèle lui-même.
    *   *Recherche d'Hyperparamètres :*
        *   Si le nombre total de combinaisons d'hyperparamètres est <= 25, GridSearchCV est utilisé.
        *   Sinon, RandomizedSearchCV est utilisé avec n_iter (minimum de 40 et le nombre total de combinaisons) et cv=3 (validation croisée à 3 plis).
        *   La métrique de scoring est r2.
    *   Le search.fit(X_train, y_train) entraîne le modèle avec recherche d'hyperparamètres.
    *   Les prédictions sont faites sur X_test.
    *   Les métriques (r2, mae, rmse) sont calculées.
*   *Résultats des Modèles (sur l'ensemble de test) :*
    | Modèle             | R2    | MAE (DH) | RMSE (DH) | Meilleurs Hyperparamètres (extrait)                                                                     |
    | ------------------ | ----- | -------- | --------- | ------------------------------------------------------------------------------------------------------- |
    | *MLP*            | 0.871 | 3143.6   | 5337.1    | solver: adam, max_iter: 300, lr_init: 0.1, lr: constant, hidden_layers: (50,), alpha: 0.0001, act: relu |
    | GradientBoosting   | 0.871 | 3151.8   | 5340.5    | subsample: 0.8, n_estimators: 200, loss: squared_error, lr: 0.2                                         |
    | RandomForest       | 0.861 | 3396.2   | 5552.3    | criterion: squared_error, max_depth: 10, n_estimators: 100                                              |
    | DecisionTree       | 0.847 | 3550.3   | 5822.8    | min_samples_split: 2, max_depth: 10, criterion: absolute_error                                          |
    | LinearRegression   | 0.777 | 4759.0   | 7032.9    | Aucun                                                                                                   |

### 5.7. Sélection et Sauvegarde du Meilleur Modèle
*   Le modèle *MLPRegressor* est sélectionné comme le meilleur modèle basé sur le score R² le plus élevé.
*   Le pipeline complet (prétraitement + modèle MLP optimisé) est sauvegardé en utilisant joblib.dump() dans le fichier modele_selection.joblib (initialement dans un sous-dossier model/).
*   Des visualisations (prédictions vs. réelles, distribution des résidus) sont générées pour le meilleur modèle.

## 6. Déploiement

### 6.1. API FastAPI (api.py)
L'API est conçue pour charger le modèle sauvegardé et fournir un endpoint pour les prédictions.

*   *Chargement du Modèle :*
    python
    model = joblib.load('modele_selection.joblib')
    
*   **Définition des Données d'Entrée (InputFeatures avec Pydantic) :**
    Définit la structure attendue pour les requêtes JSON, incluant les types de données, des exemples, et des validateurs pour les champs catégoriels afin d'assurer que les valeurs reçues sont parmi celles attendues par le modèle (ex: SEXES, LIEUX, etc.). La liste ORIGINAL_FEATURE_NAMES garantit que le DataFrame créé à partir des données d'entrée a ses colonnes dans le bon ordre avant d'être passé au pipeline de prédiction.
*   **Endpoint de Prédiction (/predict/) :**
    *   Méthode : POST
    *   Accepte un JSON correspondant au schéma InputFeatures.
    *   Convertit les données d'entrée en DataFrame Pandas, en s'assurant de l'ordre correct des colonnes.
    *   Utilise model.predict(input_df) pour obtenir la prédiction.
    *   Retourne un JSON : {"revenu_annuel_predit_dh": valeur_predite}.
    *   Gestion des erreurs : HTTPException pour modèle non chargé, erreurs de données, ou erreurs de prédiction.
*   **Endpoint Racine (/) :**
    *   Méthode : GET
    *   Retourne un message simple pour tester la disponibilité de l'API.
*   *Lancement :*
    Utiliser Uvicorn : uvicorn api:app --reload --host 127.0.0.1 --port 8000
    La documentation interactive (Swagger UI) est accessible via /docs et ReDoc via /redoc.

### 6.2. Application Web Streamlit (app.py)
L'application Streamlit fournit une interface utilisateur pour interagir avec l'API FastAPI.

*   *Interface Utilisateur :*
    *   Utilise st.sidebar pour les champs de saisie (sliders pour l'âge et l'expérience, number_input pour le nombre d'enfants, selectbox pour les variables catégorielles avec des options prédéfinies).
    *   Affiche un récapitulatif des informations saisies.
*   *Logique de Prédiction :*
    *   Lorsqu'on clique sur le bouton "Prédire le Revenu Annuel" :
        1.  Les données saisies sont collectées dans un dictionnaire.
        2.  Une requête POST est envoyée à l'URL de l'API FastAPI (API_URL) avec les données en format JSON.
        3.  La réponse de l'API est traitée.
        4.  Le revenu prédit est affiché à l'aide de st.metric.
        5.  Une comparaison indicative avec le revenu moyen national simulé est fournie.
*   *Gestion des Erreurs :*
    *   Affiche des messages d'erreur clairs en cas d'échec de connexion à l'API, de timeout, d'erreurs HTTP, ou d'autres exceptions.
*   *Lancement :*
    Exécuter : streamlit run app.py

## 7. Instructions d'Exécution

1.  *Cloner le Répertoire (si applicable) ou Placer les Fichiers :*
    Assurez-vous que IA_Projet_hala_zakaria_jihane_(3) (2).ipynb, api.py, app.py et dataset_revenu_marocains.csv (ou le script generate_dataset.py) sont dans le même répertoire de travail ou organisés de manière à ce que les chemins soient corrects.
2.  *Créer un Environnement Virtuel (Recommandé) :*
    bash
    python -m venv env_revenu
    source env_revenu/bin/activate  # Linux/macOS
    # env_revenu\Scripts\activate    # Windows
    
3.  *Installer les Dépendances :*
    bash
    pip install numpy==1.23.5 pandas matplotlib seaborn sweetviz scikit-learn joblib fastapi "uvicorn[standard]" streamlit pydantic typing-extensions
    
    (Note : typing-extensions est souvent une dépendance de Pydantic/FastAPI).
4.  *Générer le Dataset (si nécessaire) :*
    Si dataset_revenu_marocains.csv n'est pas fourni, exécutez python generate_dataset.py.
5.  *Exécuter le Notebook Jupyter :*
    *   Lancez Jupyter : jupyter notebook ou jupyter lab.
    *   Ouvrez IA_Projet_hala_zakaria_jihane_(3) (2).ipynb.
    *   Exécutez toutes les cellules. Cela va :
        *   Générer rapport_AI_analyse.html.
        *   Entraîner les modèles et sauvegarder le meilleur (pipeline complet) sous modele_selection.joblib.
    *   *Important :* Assurez-vous que modele_selection.joblib est à la racine du projet où api.py sera exécuté, ou ajustez le chemin dans api.py. Le notebook sauvegarde dans model/modele_selection.joblib.
6.  *Démarrer l'API FastAPI :*
    Dans un terminal, à la racine du projet :
    bash
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
    
    Vérifiez que l'API fonctionne en accédant à http://127.0.0.1:8000/docs dans votre navigateur.
7.  *Lancer l'Application Streamlit :*
    Dans un autre terminal, à la racine du projet :
    bash
    streamlit run app.py
    
    L'application web devrait s'ouvrir automatiquement dans votre navigateur.

## 8. Résultats Clés et Analyse

*   Le modèle *MLPRegressor* a offert la meilleure performance prédictive (R² = 0.871 sur le set de test), indiquant qu'il explique environ 87.1% de la variance du revenu annuel.
*   Le MAE de 3143.64 DH signifie qu'en moyenne, les prédictions du modèle s'écartent d'environ 3143 DH de la valeur réelle.
*   Le RMSE de 5337.09 DH, étant plus sensible aux grosses erreurs, donne une autre mesure de l'erreur type.
*   La distribution des résidus (graphique dans le notebook) semble globalement centrée autour de zéro, ce qui est un bon signe, bien qu'une analyse plus poussée de sa forme (normalité, hétéroscédasticité) pourrait être menée.
*   La suppression de Redondance_Transport et Identifiant_Aleatoire était justifiée, la première étant potentiellement redondante avec Type_Transport ou Biens_Possedes, et la seconde n'ayant aucune valeur prédictive.

## 9. Pistes d'Amélioration

*   *Ingénierie des Caractéristiques Avancée :*
    *   Créer des termes d'interaction (ex: Age * Experience_Annees).
    *   Transformer des variables numériques (ex: log(Nombre_Enfants + 1) si la distribution est très asymétrique).
    *   Encoder les variables catégorielles ordinales (comme Niveau_Education) de manière plus fine si un ordre naturel existe et est pertinent (ex: OrdinalEncoder au lieu de OneHotEncoder si les modèles d'arbres sont privilégiés).
*   *Exploration de Modèles Plus Complexes :*
    *   Tester des algorithmes de boosting plus récents et performants comme XGBoost, LightGBM, ou CatBoost.
    *   Explorer des architectures de réseaux de neurones plus profondes ou différentes pour le MLPRegressor.
*   *Optimisation des Hyperparamètres :*
    *   Utiliser des techniques de recherche plus avancées (ex: Optuna, Hyperopt).
    *   Élargir les grilles de recherche et augmenter le nombre de plis de validation croisée (ex: cv=5 ou cv=10) si les ressources le permettent.
*   *Gestion des Outliers :* Étudier des méthodes de traitement des outliers plus robustes que la simple suppression (ex: winsorisation, transformation).
*   *Interprétabilité du Modèle :* Utiliser des outils comme SHAP ou LIME pour comprendre quelles caractéristiques influencent le plus les prédictions du modèle MLP.
*   *Robustesse de l'API et de l'Application :*
    *   Ajouter une journalisation (logging) plus détaillée.
    *   Mettre en place des tests unitaires et d'intégration.
    *   Sécuriser l'API si elle devait être exposée publiquement.
    *   Utiliser Docker pour conteneuriser l'API et l'application Streamlit pour un déploiement facilité.
*   *Dataset :* Si possible, utiliser des données réelles ou affiner les contraintes de génération du dataset synthétique pour plus de réalisme.

## 10. Livrables Attendus

1.  generate_dataset.py
2.  dataset_revenu_marocains.csv
3.  IA_Projet_hala_zakaria_jihane_(3) (2).ipynb
4.  modele_selection.joblib (le pipeline modèle sauvegardé)
5.  api.py (script FastAPI)
6.  app.py (script Streamlit)
7.  README.md (ce document)
8.  Tout autre fichier nécessaire (ex: rapport_AI_analyse.html, requirements.txt).

---
Cette documentation détaillée devrait couvrir l'ensemble des aspects de notre projet.