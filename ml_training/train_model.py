from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class PatientDeteriorationModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',  # Important car données déséquilibrées
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """Prépare les features pour l'entraînement"""
        
        # Calcul des tendances sur fenêtre glissante
        df['hr_trend_1h'] = df.groupby('patient_id')['hr'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().diff()
        )
        df['spo2_trend_1h'] = df.groupby('patient_id')['spo2'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().diff()
        )
        df['bp_trend_1h'] = df.groupby('patient_id')['bp'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().diff()
        )
        
        # Features de variabilité
        df['hr_variability'] = df.groupby('patient_id')['hr'].transform(
            lambda x: x.rolling(window=8, min_periods=1).std()
        )
        
        # Features d'interaction
        df['hr_spo2_ratio'] = df['hr'] / df['spo2']
        df['temp_bp_product'] = df['temp'] * df['bp'] / 100
        
        # Sélection des features
        feature_cols = [
            'hr', 'spo2', 'bp', 'temp', 'rr', 'age',
            'hr_trend_1h', 'spo2_trend_1h', 'bp_trend_1h',
            'hr_variability', 'hr_spo2_ratio', 'temp_bp_product'
        ]
        
        # Remplir les NaN
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        self.feature_names = feature_cols
        return df[feature_cols], df['deterioration']
    
    def train(self, data_path='patient_history.csv'):
        """Entraîne le modèle"""
        
        print("📊 Chargement des données...")
        df = pd.read_csv(data_path)
        print(f"   {len(df)} enregistrements chargés")
        
        # Préparation des features
        X, y = self.prepare_features(df)
        print(f"   Features utilisées : {list(X.columns)}")
        print(f"   Ratio détérioration : {y.mean():.2%}")
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("🧠 Entraînement du modèle Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Évaluation
        print("📈 Évaluation du modèle...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*50)
        print("📊 RAPPORT DE CLASSIFICATION")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        print(f"\n📊 MATRICE DE CONFUSION")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"\n🎯 AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='roc_auc')
        print(f"\n🔍 Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 IMPORTANCE DES FEATURES")
        print(feature_importance.head(10))
        
        return self
    
    def save(self, model_path='deterioration_model.pkl'):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, model_path)
        print(f"💾 Modèle sauvegardé : {model_path}")
    
    @staticmethod
    def load(model_path='deterioration_model.pkl'):
        """Charge un modèle sauvegardé"""
        data = joblib.load(model_path)
        model_obj = PatientDeteriorationModel()
        model_obj.model = data['model']
        model_obj.scaler = data['scaler']
        model_obj.feature_names = data['feature_names']
        print(f"📂 Modèle chargé : {model_path}")
        return model_obj

if __name__ == "__main__":
    # Générer les données si elles n'existent pas
    try:
        pd.read_csv('patient_history.csv')
    except:
        print("⚠️  Données non trouvées, génération...")
        from generate_training_data import generate_training_data
        df = generate_training_data()
        df.to_csv('patient_history.csv', index=False)
    
    # Entraîner et sauvegarder le modèle
    trainer = PatientDeteriorationModel()
    trainer.train()
    trainer.save()
    
    # Test de prédiction
    print("\n🧪 TEST DE PRÉDICTION")
    test_features = np.array([[75, 97, 120, 36.8, 16, 65, 0.5, -0.2, 1.0, 3.5, 0.77, 44.16]])
    loaded_model = PatientDeteriorationModel.load()
    
    # Préparation pour prédiction
    test_scaled = loaded_model.scaler.transform(test_features)
    proba = loaded_model.model.predict_proba(test_scaled)[0][1]
    
    print(f"Risque de détérioration : {proba*100:.1f}%")
    print(f"Niveau de risque : {'CRITIQUE' if proba > 0.7 else 'ÉLEVÉ' if proba > 0.5 else 'MODÉRÉ' if proba > 0.3 else 'FAIBLE'}")