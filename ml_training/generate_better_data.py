import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_deterioration_data():
    """Génère des données avec des détériorations réalistes"""
    
    records = []
    
    # Scénario 1: Détérioration rapide (sepsis)
    for i in range(24):  # 24 heures
        # Détérioration entre heures 12-18
        if 12 <= i < 18:
            hr = 80 + (i-12)*10  # FC monte de 80 à 140
            spo2 = 98 - (i-12)*2  # SpO2 descend de 98 à 86
            temp = 37.0 + (i-12)*0.5  # Temp monte à 40
            deterioration = 1 if i >= 15 else 0  # Détérioration après 3h
        else:
            hr = 75 + np.random.normal(0, 5)
            spo2 = 97 + np.random.normal(0, 1)
            temp = 36.8 + np.random.normal(0, 0.2)
            deterioration = 0
        
        records.append({
            'hr': hr,
            'spo2': spo2,
            'bp': 120 + np.random.normal(0, 10),
            'temp': temp,
            'rr': 16 + np.random.normal(0, 2),
            'age': 65,
            'deterioration': deterioration
        })
    
    # Scénario 2: Détérioration lente (insuffisance respiratoire)
    for i in range(48):  # 48 heures
        if i > 24:
            spo2 = 97 * (0.99 ** (i-24))  # Décroissance exponentielle
            deterioration = 1 if spo2 < 92 else 0
        else:
            spo2 = 97 + np.random.normal(0, 1)
            deterioration = 0
        
        records.append({
            'hr': 85 + np.random.normal(0, 8),
            'spo2': spo2,
            'bp': 130 + np.random.normal(0, 12),
            'temp': 37.2 + np.random.normal(0, 0.3),
            'rr': 18 + np.random.normal(0, 3),
            'age': 72,
            'deterioration': deterioration
        })
    
    df = pd.DataFrame(records)
    
    # Calculer les tendances (comme le fera votre API)
    df['hr_trend_1h'] = df['hr'].diff(4).fillna(0) / 4
    df['spo2_trend_1h'] = df['spo2'].diff(4).fillna(0) / 4
    df['bp_trend_1h'] = df['bp'].diff(4).fillna(0) / 4
    df['hr_variability'] = df['hr'].rolling(4).std().fillna(0)
    df['hr_spo2_ratio'] = df['hr'] / df['spo2']
    df['temp_bp_product'] = df['temp'] * df['bp'] / 100
    
    return df

if __name__ == "__main__":
    print("🔄 Génération de données réalistes...")
    df = generate_realistic_deterioration_data()
    
    print(f"📊 {len(df)} enregistrements")
    print(f"🎯 Détériorations: {df['deterioration'].sum()} ({df['deterioration'].mean():.1%})")
    
    # Afficher un exemple de détérioration
    print("\n📈 Exemple de détérioration:")
    sample = df[df['deterioration'] == 1].head(3)
    for _, row in sample.iterrows():
        print(f"  HR: {row['hr']:.0f}, SpO2: {row['spo2']:.1f}, Temp: {row['temp']:.1f}")
        print(f"  Trends: HR:{row['hr_trend_1h']:.1f}, SpO2:{row['spo2_trend_1h']:.1f}")
    
    # Sauvegarder
    df.to_csv('realistic_patient_data.csv', index=False)
    print(f"\n💾 Données sauvegardées: realistic_patient_data.csv")