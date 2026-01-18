from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes depuis Node-RED

# ============================================
# CONFIGURATION MODÈLE ML
# ============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'deterioration_model.pkl')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    print(f"✅ Modèle chargé avec {len(feature_names)} features")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = scaler = feature_names = None

# ============================================
# FONCTION PRÉPARATION DONNÉES
# ============================================

def prepare_single_prediction(features_dict):
    """Prépare les features pour une prédiction unique"""
    
    print(f"\n📥 Données reçues pour prédiction ML:")
    print(f"  HR: {features_dict.get('hr', 'N/A')}, SpO2: {features_dict.get('spo2', 'N/A')}")
    
    # VALIDATION DES DONNÉES
    hr = float(features_dict.get('hr', 75))
    spo2 = float(features_dict.get('spo2', 97))
    bp = float(features_dict.get('bp', 120))
    temp = float(features_dict.get('temp', 36.8))
    rr = float(features_dict.get('rr', 16))
    age = float(features_dict.get('age', 50))
    
    # Corriger les valeurs aberrantes
    if hr < 30 or hr > 200: hr = 75
    if spo2 < 70 or spo2 > 100: spo2 = 97
    if bp < 60 or bp > 200: bp = 120
    if temp < 30 or temp > 42: temp = 36.8
    if rr < 8 or rr > 40: rr = 16
    if age < 0 or age > 120: age = 50
    
    # CALCUL DES TENDANCES
    hr_trend_1h = spo2_trend_1h = bp_trend_1h = hr_variability = 0
    
    if 'history' in features_dict and len(features_dict['history']) >= 4:
        history = features_dict['history']
        recent = history[-4:]
        
        if all('hr' in h for h in recent):
            hr_trend_1h = (recent[-1]['hr'] - recent[0]['hr']) / 3
            hrs = [h['hr'] for h in recent]
            hr_variability = np.std(hrs) if len(hrs) > 1 else 0
            
        if all('spo2' in h for h in recent):
            spo2_trend_1h = (recent[-1]['spo2'] - recent[0]['spo2']) / 3
            
        if all('bp' in h for h in recent):
            bp_trend_1h = (recent[-1]['bp'] - recent[0]['bp']) / 3
    
    # Construction des features
    base_features = [
        float(hr),
        float(spo2),
        float(bp),
        float(temp),
        float(rr),
        float(age),
        float(hr_trend_1h),
        float(spo2_trend_1h),
        float(bp_trend_1h),
        float(hr_variability),
        float(hr) / max(float(spo2), 1),
        float(temp) * float(bp) / 100
    ]
    
    return np.array(base_features).reshape(1, -1)

# ============================================
# ROUTES API
# ============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'API ML pour monitoring hospitalier',
        'model_loaded': model is not None,
        'endpoints': {
            '/predict': 'POST - Prédire risque détérioration',
            '/batch_predict': 'POST - Prédictions multiples',
            '/model_info': 'GET - Informations modèle',
            '/health': 'GET - Statut API'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    if not model:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_names),
        'features': feature_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API de prédiction pure - PAS de publication"""
    if not model:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.json
        
        print(f"\n{'='*60}")
        print(f"🎯 PRÉDICTION ML SEULEMENT")
        print(f"{'='*60}")
        
        # Format 1: Données brutes
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        
        # Format 2: Données structurées
        elif all(key in data for key in ['hr', 'spo2', 'bp', 'temp', 'rr']):
            features = prepare_single_prediction(data)
        
        else:
            return jsonify({'error': 'Format de données non reconnu'}), 400
        
        # Vérifier la shape
        if features.shape[1] != len(feature_names):
            return jsonify({
                'error': f'Mismatch features. Reçu {features.shape[1]}, modèle attend {len(feature_names)}'
            }), 400
        
        # Convertir en DataFrame
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Normalisation
        features_scaled = scaler.transform(features_df)
        
        # Prédiction
        prediction_proba = model.predict_proba(features_scaled)[0]
        ml_score = float(prediction_proba[1] * 100)
        
        print(f"📈 Score ML pur: {ml_score:.2f}%")
        
        # Règles médicales NEWS2
        medical_score = 0
        
        if data.get('spo2', 100) <= 91:
            medical_score += 35
        elif data.get('spo2', 100) <= 93:
            medical_score += 25
        elif data.get('spo2', 100) <= 95:
            medical_score += 15

        if data.get('hr', 0) >= 131 or data.get('hr', 0) <= 40:
            medical_score += 30
        elif data.get('hr', 0) >= 111 or data.get('hr', 0) <= 50:
            medical_score += 20

        if data.get('temp', 0) >= 39.1:
            medical_score += 25
        elif data.get('temp', 0) >= 38.1:
            medical_score += 15

        if data.get('rr', 0) >= 25 or data.get('rr', 0) <= 8:
            medical_score += 25
        elif data.get('rr', 0) >= 21 or data.get('rr', 0) <= 11:
            medical_score += 15
        
        # Score final combiné
        final_score = max(ml_score, medical_score)
        
        # Détermination niveau
        if final_score >= 80:
            risk_level = "CRITIQUE"
            recommendation = "Intervention médicale immédiate"
        elif final_score >= 60:
            risk_level = "ÉLEVÉ"
            recommendation = "Surveillance continue, alerte équipe"
        elif final_score >= 40:
            risk_level = "MODÉRÉ"
            recommendation = "Vérification paramètres, surveillance renforcée"
        elif final_score >= 20:
            risk_level = "ATTENTION"
            recommendation = "Surveillance standard"
        else:
            risk_level = "FAIBLE"
            recommendation = "Surveillance normale"
        
        print(f"🎯 Score final: {final_score:.2f}% - Niveau: {risk_level}")
        
        # Résultat SANS publication
        result = {
            'risk_score': final_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'ml_score': ml_score,
            'medical_rules_score': medical_score,
            'medical_override_applied': medical_score > ml_score,
            'timestamp': datetime.now().isoformat(),
            'patient_id': data.get('patient_id', 'unknown'),
            'age': data.get('age', 0),
            'hr': data.get('hr', 0),
            'spo2': data.get('spo2', 0),
            'bp': data.get('bp', 0),
            'temp': data.get('temp', 0),
            'rr': data.get('rr', 0),
            'published_by': 'node_red'  # Indique que Node-RED fera la publication
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur ML: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch predictions - PAS de publication"""
    if not model:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.json
        patients = data.get('patients', [])
        
        print(f"\n👥 BATCH PREDICTION: {len(patients)} patients")
        
        results = []
        for patient in patients:
            features = prepare_single_prediction(patient)
            features_df = pd.DataFrame(features, columns=feature_names)
            features_scaled = scaler.transform(features_df)
            
            prediction_proba = model.predict_proba(features_scaled)[0]
            ml_score = float(prediction_proba[1] * 100)
            
            # Règles médicales simplifiées
            medical_score = 0
            if patient.get('spo2', 100) <= 91: medical_score += 35
            elif patient.get('spo2', 100) <= 93: medical_score += 25
            elif patient.get('spo2', 100) <= 95: medical_score += 15
            
            if patient.get('hr', 0) >= 131 or patient.get('hr', 0) <= 40: medical_score += 30
            elif patient.get('hr', 0) >= 111 or patient.get('hr', 0) <= 50: medical_score += 20
            
            final_score = max(ml_score, medical_score)
            
            # Niveau de risque
            if final_score >= 70:
                risk_level = "CRITIQUE"
            elif final_score >= 50:
                risk_level = "ÉLEVÉ"
            elif final_score >= 30:
                risk_level = "MODÉRÉ"
            else:
                risk_level = "FAIBLE"
            
            results.append({
                'patient_id': patient.get('patient_id', 'unknown'),
                'risk_score': final_score,
                'risk_level': risk_level,
                'ml_score': ml_score,
                'medical_score': medical_score
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    return jsonify({'error': 'Modèle non trouvé'}), 404

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 API ML - MONITORING HOSPITALIER")
    print("="*60)
    print(f"📊 Modèle: {'✅ Chargé' if model else '❌ Non chargé'}")
    print(f"🎯 Fonction: Prédiction ML uniquement")
    print(f"📤 Publication: Gérée par Node-RED")
    print(f"🔗 API: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)