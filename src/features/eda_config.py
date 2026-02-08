"""
EDA-Konfiguration für Fraud Detection
======================================
Alle wichtigen Erkenntnisse aus der explorativen Datenanalyse (03_eda.ipynb)
für die Verwendung in der Modellierung (04_modeling.ipynb).

Generiert am: 2026-02-06
"""

# ============================================================================
# FEATURE-AUSWAHL
# ============================================================================

# ✅ Features, die für Modellierung verwendet werden sollen
FEATURES_TO_USE = [
    # Numerische Features
    'log_value',           # 600x Risikofaktor (nach Schwellenwert-Transformation)
    'ts_month',            # 1.66x leichter saisonaler Effekt
    
    # Zyklische Zeit-Features
    'ts_hour_sin',         # Stunden-Zyklus (sin-Komponente)
    'ts_hour_cos',         # Stunden-Zyklus (cos-Komponente)
    'ts_dow_sin',          # Wochentag-Zyklus (sin-Komponente)
    'ts_dow_cos',          # Wochentag-Zyklus (cos-Komponente)
    
    # Kategoriale Features
    'ProductId',           # 87x Risikofaktor - BESTES FEATURE
    'PricingStrategy',     # 46x Risikofaktor
    'ProductCategory',     # 40x Risikofaktor
    'ProviderId',          # 10x Risikofaktor
    'ChannelId',           # 3.7x Risikofaktor
]

# ❌ Features, die NICHT verwendet werden sollen (redundant oder nutzlos)
FEATURES_TO_DROP = [
    'Amount',              # Redundant mit log_value (0.99 Korrelation)
    'Value',               # Redundant mit log_value (0.99 Korrelation)
    'ts_is_night',         # Kein Effekt (0.88x Risikofaktor)
    'ts_is_weekend',       # Kein Effekt (1.00x Risikofaktor)
    'amount_value_ratio',  # Nutzlos (100% haben Abweichungen)
]

# Kategoriale Features (benötigen Encoding)
CATEGORICAL_FEATURES = [
    'ProviderId',
    'ProductId',
    'ProductCategory',
    'ChannelId',
    'PricingStrategy'
]

# Numerische Features (benötigen ggf. Scaling)
NUMERIC_FEATURES = [
    'log_value',
    'ts_month',
    'ts_hour_sin',
    'ts_hour_cos',
    'ts_dow_sin',
    'ts_dow_cos'
]

# Zielvariable
TARGET = 'FraudResult'

# ============================================================================
# SCHWELLENWERTE & FEATURE ENGINEERING
# ============================================================================

# Schwellenwert für "hohe Beträge" (95% Quantil der normalen Transaktionen)
HIGH_AMOUNT_THRESHOLD = 12500.0  # €

# Neues Feature: is_high_amount
# Verwendung: df['is_high_amount'] = (df['Amount'] > HIGH_AMOUNT_THRESHOLD).astype(int)

# ============================================================================
# RISIKOFAKTOREN (für Dokumentation)
# ============================================================================

RISK_FACTORS = {
    'log_value': 600.0,           # Nach Schwellenwert-Transformation
    'ProductId': 87.0,
    'PricingStrategy': 46.0,
    'ProductCategory': 40.0,
    'ProviderId': 10.0,
    'ChannelId': 3.7,
    'ts_month': 1.66,
    'is_high_amount': 18.9,       # Neu zu erstellen
}

# ============================================================================
# KLASSENVERTEILUNG & BASELINE
# ============================================================================

# Klassenimbalance
CLASS_DISTRIBUTION = {
    'normal': 0.998,      # 99.8% normale Transaktionen
    'fraud': 0.002,       # 0.2% Betrug
}

# Baseline Accuracy (immer "kein Betrug" vorhersagen)
BASELINE_ACCURACY = 0.998
BASELINE_RECALL = 0.0         # Findet KEINEN Betrug!

# Wichtig: Accuracy ist irreführend bei unausgeglichenen Daten
# → Verwende Recall, Precision, F1-Score, AUC-ROC

# ============================================================================
# MODELLIERUNGS-EMPFEHLUNGEN
# ============================================================================

RECOMMENDED_MODELS = [
    'XGBoost',            # Beste Wahl für unausgeglichene Daten + Interaktionen
    'RandomForest',       # Robust, lernt Interaktionen automatisch
    'LightGBM',           # Schneller als XGBoost, ähnliche Performance
]

# Nicht empfohlen:
# - Logistische Regression (erkennt keine Interaktionen, nicht für non-lineare Schwellenwerte)
# - Naive Bayes (falsche Unabhängigkeitsannahmen)

# ============================================================================
# WICHTIGE ERKENNTNISSE
# ============================================================================

KEY_INSIGHTS = {
    'threshold_effect': 'Beträge > 12.500€ haben 18.9x höheres Betrugsrisiko',
    'best_features': 'ProductId (87x), PricingStrategy (46x), ProductCategory (40x)',
    'redundancy': 'Amount und Value zu 99% korreliert → nur log_value verwenden',
    'time_effect': 'Tageszeit (night/weekend) hat KEINEN Effekt',
    'interactions': 'ProviderId × ProductId zeigt bis zu 18% Betrugsrate',
    'class_imbalance': '0.2% Betrug → SMOTE/Klassen-Gewichtung empfohlen',
}

# ============================================================================
# VORVERARBEITUNGSSCHRITTE FÜR MODELLIERUNG
# ============================================================================

PREPROCESSING_STEPS = """
1. Features auswählen:
   - Verwende: FEATURES_TO_USE
   - Entferne: FEATURES_TO_DROP

2. Neues Feature erstellen:
   df['is_high_amount'] = (df['Amount'] > HIGH_AMOUNT_THRESHOLD).astype(int)

3. Kategoriale Features encoden:
   - Für Baummodelle: Label Encoding oder direkt als Kategorie
   - Für lineare Modelle: One-Hot Encoding (aber nicht empfohlen)

4. Numerische Features skalieren (optional für XGBoost/RF):
   - StandardScaler oder MinMaxScaler
   - Nur wenn lineare Modelle verwendet werden

5. Train-Test-Split:
   - Stratifiziert nach FraudResult (erhält Klassenverteilung)
   - test_size=0.2 oder 0.25

6. Klassenimbalance behandeln:
   - Option 1: class_weight='balanced' (XGBoost: scale_pos_weight)
   - Option 2: SMOTE (Synthetic Minority Over-sampling)
   - Option 3: Untersampling der Mehrheitsklasse

7. Kreuzvalidierung:
   - StratifiedKFold (5-10 Folds)
   - Erhält Klassenverteilung in jedem Fold
"""

# ============================================================================
# METRIKEN FÜR EVALUATION
# ============================================================================

EVALUATION_METRICS = [
    'recall',              # Wie viele Betrüger finden wir? (WICHTIGSTE METRIK)
    'precision',           # Wie viele Vorhersagen sind korrekt?
    'f1_score',            # Balance zwischen Recall und Precision
    'roc_auc',             # Gesamtleistung über alle Schwellenwerte
    'pr_auc',              # Präzision-Recall-AUC (besser für unausgeglichene Daten)
]

# Nicht verwenden: Accuracy (irreführend bei 99.8% Normalfällen)

# ============================================================================
# ZIEL-METRIKEN
# ============================================================================

TARGET_METRICS = {
    'min_recall': 0.7,        # Mindestens 70% der Betrüger finden
    'min_precision': 0.1,     # Maximal 10 Fehlalarme pro echter Betrug
    'min_f1': 0.15,           # Gute Balance
    'min_roc_auc': 0.85,      # Sehr gute Gesamtleistung
}

# Business-Kontext:
# - Hoher Recall wichtiger als hohe Precision
# - Lieber 10 Fehlalarme als 1 übersehenen Betrug
# - Recall > 0.7 ist realistisches Ziel

# ============================================================================
# DATENPFADE
# ============================================================================

DATA_PATH = "../data/processed/training_preprocessed.csv"

# ============================================================================
# HYPERPARAMETER-EMPFEHLUNGEN
# ============================================================================

# XGBoost Hyperparameter (Startpunkt für Tuning)
XGBOOST_PARAMS = {
    'max_depth': 6,                    # Tiefe der Bäume
    'learning_rate': 0.1,              # Lernrate
    'n_estimators': 100,               # Anzahl Bäume
    'subsample': 0.8,                  # Anteil Samples pro Baum
    'colsample_bytree': 0.8,           # Anteil Features pro Baum
    'min_child_weight': 1,             # Minimum Gewicht in Blatt
    'gamma': 0,                        # Minimum Loss Reduction
    'random_state': 42,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 10,       # Stoppe wenn keine Verbesserung
}

# RandomForest Hyperparameter
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
}

# ============================================================================
# TOP FEATURE-KOMBINATIONEN (aus EDA)
# ============================================================================

# Riskanteste Provider-Produkt-Kombinationen (Betrugsrate > 5%)
HIGH_RISK_COMBINATIONS = [
    ('ProviderId_5', 'ProductId_9'),    # 18.75% Betrugsrate
    ('ProviderId_1', 'ProductId_5'),    # 18.18% Betrugsrate
    ('ProviderId_5', 'ProductId_22'),   # 10.00% Betrugsrate
    ('ProviderId_5', 'ProductId_13'),   # 7.06% Betrugsrate
]

# ============================================================================
# CROSS-VALIDATION STRATEGIE
# ============================================================================

CV_STRATEGY = {
    'method': 'StratifiedKFold',
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
}

# ============================================================================
# WICHTIGE SCHWELLENWERTE AUS EDA
# ============================================================================

THRESHOLDS = {
    'high_amount': 12500.0,           # 95% Quantil normale Transaktionen
    'fraud_median': 600000.0,         # Median Betrugstransaktionen
    'normal_median': 1000.0,          # Median normale Transaktionen
}

# ============================================================================
# ENCODING-STRATEGIE
# ============================================================================

ENCODING_STRATEGY = {
    'categorical': 'LabelEncoder',     # Für XGBoost/RandomForest/LightGBM
    # Alternative: 'OrdinalEncoder' oder 'OneHotEncoder' (nicht empfohlen wegen Dimensionalität)
}

# ============================================================================
# TRAIN-TEST-SPLIT
# ============================================================================

TRAIN_TEST_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,                  # Wichtig: Erhält Klassenverteilung
}

# ============================================================================
# UMGANG MIT KLASSENIMBALANCE
# ============================================================================

IMBALANCE_STRATEGY = {
    'method': 'class_weight',          # Empfohlen: class_weight='balanced'
    # Alternativen:
    # 'SMOTE': für Oversampling
    # 'RandomUnderSampler': für Undersampling
    # 'scale_pos_weight': für XGBoost (Wert: count_0 / count_1)
}

# Berechnung scale_pos_weight für XGBoost:
# scale_pos_weight = (Anzahl normal) / (Anzahl fraud) ≈ 95263 / 193 ≈ 493

SCALE_POS_WEIGHT = 493.0              # Vorberechnet aus EDA

# ============================================================================
# ERWARTETE FEATURE IMPORTANCE (Top 5)
# ============================================================================

EXPECTED_FEATURE_IMPORTANCE = [
    ('ProductId', 'Höchste Wichtigkeit'),
    ('PricingStrategy', 'Sehr hoch'),
    ('log_value / is_high_amount', 'Sehr hoch'),
    ('ProductCategory', 'Hoch'),
    ('ProviderId', 'Mittel-Hoch'),
]

# ============================================================================
# ZUSAMMENFASSUNG FÜR SCHNELLEN START
# ============================================================================

QUICK_START_CONFIG = {
    'model': 'XGBoost',
    'features': 'FEATURES_TO_USE + [is_high_amount]',
    'encoding': 'LabelEncoder für kategoriale Features',
    'imbalance': 'scale_pos_weight=493',
    'cv': 'StratifiedKFold(5)',
    'metrics': 'Recall (wichtigste), F1, ROC-AUC',
    'target': 'Recall > 0.7, F1 > 0.15',
}
