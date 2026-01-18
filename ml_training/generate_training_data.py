import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_training_data(num_patients=50, hours=72):
    """Génère des données de simulation pour l'entraînement"""
    
    records = []
    
    for patient_id in range(num_patients):
        # Profil de base aléatoire
        age = np.random.randint(20, 90)
        base_hr = np.random.normal(75, 10)
        base_spo2 = np.random.normal(97, 1.5)
        base_temp = np.random.normal(36.8, 0.5)
        
        # Simuler 72 heures de données (toutes les 15 min)
        current_time = datetime.now() - timedelta(hours=hours)
        
        for i in range(hours * 4):  # 4 mesures par heure
            # Variation naturelle
            time_factor = np.sin(i / 20) * 0.5 + 1
            
            hr = max(40, min(180, base_hr * time_factor + np.random.normal(0, 5)))
            spo2 = max(85, min(100, base_spo2 / time_factor + np.random.normal(0, 1)))
            temp = max(35, min(41, base_temp * time_factor + np.random.normal(0, 0.3)))
            bp = np.random.normal(120, 15)
            rr = np.random.normal(16, 4)
            
            # Calcul des tendances (sur les dernières 4 mesures)
            hr_trend = 0
            spo2_trend = 0
            
            # Simuler une détérioration pour certains patients
            deterioration = 0
            if i > 100:  # Après un certain temps
                # 20% de chance de détérioration
                if np.random.random() < 0.2:
                    hr *= 1.3
                    spo2 *= 0.9
                    temp += 1.5
                    deterioration = 1
            
            records.append({
                'patient_id': f'P{patient_id:03d}',
                'timestamp': current_time.isoformat(),
                'hr': round(hr, 1),
                'spo2': round(spo2, 1),
                'bp': round(bp, 1),
                'temp': round(temp, 1),
                'rr': round(rr, 1),
                'hr_trend': hr_trend,
                'spo2_trend': spo2_trend,
                'deterioration': deterioration,
                'age': age
            })
            
            current_time += timedelta(minutes=15)
    
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Génération des données d'entraînement...")
    df = generate_training_data(num_patients=30, hours=48)
    df.to_csv('patient_history.csv', index=False)
    print(f"✅ Données générées : {len(df)} enregistrements")
    print(f"    Détériorations : {df['deterioration'].sum()} cas")
    print(df.head())