import random

def kategori_tekanan_darah(sistolik, diastolik):
    if sistolik < 120 and diastolik < 80:
        return 'Normal'
    elif 120 <= sistolik <= 139 or 80 <= diastolik <= 89:
        return 'Pra-hipertensi'
    elif 140 <= sistolik <= 159 or 90 <= diastolik <= 99:
        return 'Hipertensi Derajat 1'
    elif 160 <= sistolik <= 179 or 100 <= diastolik <= 109:
        return 'Hipertensi Derajat 2'
    else:
        return 'Hipertensi Derajat 3'

def hitung_imt(berat_badan, tinggi_badan):
    return berat_badan / ((tinggi_badan / 100) ** 2)

# --- Scoring functions for the 4 specified diseases ---
def skor_koroner(usia, jenis_kelamin, sistolik, diastolik, kolesterol, gula_darah, nyeri_dada, denyut_jantung, imt, riwayat_keluarga):
    # Retained from previous version (with adjustments)
    skor = 0
    if usia > 50: skor += 1
    if jenis_kelamin == 'L': skor += 1
    if sistolik > 140: skor += 1
    if diastolik > 90: skor += 1
    if kolesterol > 200: skor += 1
    if gula_darah > 140: skor += 1
    if nyeri_dada == 'Ya': skor += 1 # Adjusted score
    if denyut_jantung > 100: skor += 1
    if imt > 30: skor += 1
    if riwayat_keluarga == 'Ya': skor += 1 # Adjusted score
    return skor

def skor_gagal_jantung(usia, sistolik, diastolik, sesak_napas, kelelahan, denyut_jantung, imt):
    # Retained from previous version
    skor = 0
    if usia > 60: skor += 1
    if sistolik > 160: skor += 1
    if diastolik > 100: skor += 1
    if sesak_napas == 'Ya': skor += 2
    if kelelahan == 'Ya': skor += 2
    if denyut_jantung < 60: skor += 1
    if imt > 30: skor += 1
    return skor

def skor_jantung_bawaan_biru_riwayat(usia, sesak_napas, imt, riwayat_keluarga):
    # New scoring function
    skor = 0
    if riwayat_keluarga == 'Ya': skor += 3 # Strongest proxy available
    if sesak_napas == 'Ya': skor += 2    # Key symptom if residual effects
    if usia < 40: skor += 1             # More likely a known condition if younger in this adult dataset
    if imt < 20 : skor += 1             # Proxy for potential past developmental issues
    return skor

def skor_demam_reumatik_dampak(usia, nyeri_dada, sesak_napas, kelelahan, denyut_jantung):
    # New scoring function
    skor = 0
    if nyeri_dada == 'Ya': skor += 2      # Cardiac involvement
    if sesak_napas == 'Ya': skor += 2    # Cardiac involvement
    if kelelahan == 'Ya': skor += 1
    if denyut_jantung > 100 or denyut_jantung < 60: skor += 2 # Valve issues, arrhythmias
    if usia < 55: skor += 1             # Long-term effects manifesting
    return skor
# --- End of scoring functions ---

def pilih_penyakit_jantung_baru(scores_dict, current_counts, target_counts):
    # This function's logic remains largely the same as the previous rewrite,
    # but operates on the new set of diseases.
    high_score_threshold = 3 # Minimum score to be considered a strong candidate
    
    # Filter for diseases that meet the high score threshold
    primary_candidates = {disease: score for disease, score in scores_dict.items() if score >= high_score_threshold}

    if not primary_candidates: # No disease scored high enough
        # Prioritize 'Tidak Ada' if it's needed
        if current_counts.get('Tidak Ada', 0) < target_counts.get('Tidak Ada', 0):
            return 'Tidak Ada'
        # If 'Tidak Ada' target is met, pick a needed disease randomly from other categories
        needed_other_diseases = [
            d for d, t_count in target_counts.items() 
            if d != 'Tidak Ada' and current_counts.get(d, 0) < t_count
        ]
        if needed_other_diseases:
            return random.choice(needed_other_diseases) 
        return 'Tidak Ada' # Fallback if all targets met or no other needs

    # Sort high-scoring candidates by score (descending)
    sorted_candidates = sorted(primary_candidates.items(), key=lambda item: item[1], reverse=True)

    # Attempt to select a high-scoring disease if it's still under its target count
    for disease, score in sorted_candidates:
        if current_counts.get(disease, 0) < target_counts.get(disease, 0):
            return disease

    # If all high-scoring candidates have met their targets, or to ensure diversity,
    # use a weighted random choice based on "need" (how far from target) and scores.
    weighted_selection_candidates = []
    for disease_name in target_counts.keys(): # Iterate over all possible outcomes including 'Tidak Ada'
        need = target_counts[disease_name] - current_counts.get(disease_name, 0)
        weight = max(0, need) + 0.5 # Base weight for being needed, plus a small constant

        if disease_name != 'Tidak Ada':
            # Add a bonus based on the actual score for this patient (normalized)
            weight += scores_dict.get(disease_name, 0) / 2.0 
        elif disease_name == 'Tidak Ada' and not primary_candidates: 
            # Boost 'Tidak Ada' if no specific disease scored high for this patient
             weight += 1.5 # Increased boost for 'Tidak Ada' when no clear disease

        # Consider for selection if needed or not excessively over target
        if current_counts.get(disease_name, 0) <= target_counts[disease_name] * 1.25 or need > 0 :
             weighted_selection_candidates.append((disease_name, weight))
    
    if not weighted_selection_candidates:
        # Fallback: if all candidates are filtered (e.g. all excessively over target)
        if sorted_candidates: # Pick highest score from original primary list
            return sorted_candidates[0][0]
        if current_counts.get('Tidak Ada', 0) < target_counts.get('Tidak Ada', 0):
            return 'Tidak Ada'
        return random.choice(list(target_counts.keys())) # Absolute fallback

    total_weight = sum(w for _, w in weighted_selection_candidates)
    if total_weight <= 0:
        if current_counts.get('Tidak Ada', 0) < target_counts.get('Tidak Ada', 0):
            return 'Tidak Ada'
        if sorted_candidates:
             max_s = sorted_candidates[0][1]
             tied_s = [d for d,s in sorted_candidates if s == max_s]
             return random.choice(tied_s)
        return random.choice(list(target_counts.keys()))

    r = random.uniform(0, total_weight)
    upto = 0
    for disease, weight_val in weighted_selection_candidates:
        if upto + weight_val >= r:
            return disease
        upto += weight_val
    
    return weighted_selection_candidates[-1][0] if weighted_selection_candidates else random.choice(list(target_counts.keys()))


def generate_dummy_data_baru(num_records=300):
    header = "Nama;Usia;Jenis Kelamin;Sistolik;Diastolik;Tekanan Darah;Kategori Tekanan Darah;Kolesterol;Gula Darah;Nyeri Dada;Sesak Napas;Kelelahan;Denyut Jantung;Penyakit Jantung"
    data = [header]

    disease_proportions = {
        'Penyakit jantung koroner': 0.20,
        'Jantung bawaan biru (Riwayat)': 0.20,
        'Gagal jantung': 0.20,
        'Demam reumatik (Dampak)': 0.20,
        'Tidak Ada': 0.20 
    }
    
    target_counts = {disease: int(num_records * prop) for disease, prop in disease_proportions.items()}
    
    current_sum = sum(target_counts.values())
    diff = num_records - current_sum
    if diff != 0: # Adjust one category (e.g., 'Tidak Ada') to make sum exactly num_records
        target_counts['Tidak Ada'] = target_counts.get('Tidak Ada',0) + diff

    current_disease_counts = {disease: 0 for disease in target_counts.keys()}

    for i in range(1, num_records + 1):
        nama = f"Pasien {i}"
        usia = random.randint(30, 70) # Original age range
        jenis_kelamin = random.choice(['L', 'P'])

        # Patient attribute generation (same as before)
        if usia < 45:
            sistolik = random.randint(110, 140)
            diastolik = random.randint(70, 90)
        elif usia < 60:
            sistolik = random.randint(120, 160)
            diastolik = random.randint(80, 100)
        else:
            sistolik = random.randint(130, 190)
            diastolik = random.randint(85, 120)
        
        tekanan_darah = f"{sistolik}/{diastolik}"
        tekanan_cat = kategori_tekanan_darah(sistolik, diastolik)

        if usia < 45:
            kolesterol = random.randint(170, 220)
        elif usia < 60:
            kolesterol = random.randint(190, 260)
        else:
            kolesterol = random.randint(200, 280)

        if tekanan_cat.startswith('Hipertensi'):
            gula_darah = random.randint(110, 180)
        else:
            gula_darah = random.randint(90, 150)

        denyut_jantung = random.randint(50, 120)
        berat_badan = random.randint(50, 100)
        tinggi_badan = random.randint(150, 180)
        imt = hitung_imt(berat_badan, tinggi_badan)

        temp_score = (usia > 50) + (sistolik > 140) + (kolesterol > 200) + (imt > 30)
        riwayat_keluarga = 'Ya' if random.random() < (0.2 + temp_score * 0.03) else 'Tidak'
        
        temp_score += (gula_darah > 140) + (riwayat_keluarga == 'Ya')
        nyeri_dada = 'Ya' if random.random() < (0.2 + temp_score * 0.03) else 'Tidak'
        sesak_napas = 'Ya' if random.random() < (0.15 + temp_score * 0.03) else 'Tidak'
        kelelahan = 'Ya' if random.random() < (0.25 + temp_score * 0.03) else 'Tidak'

        # Calculate scores for the 4 specified diseases
        all_scores = {
            'Penyakit jantung koroner': skor_koroner(usia, jenis_kelamin, sistolik, diastolik, kolesterol, gula_darah, nyeri_dada, denyut_jantung, imt, riwayat_keluarga),
            'Gagal jantung': skor_gagal_jantung(usia, sistolik, diastolik, sesak_napas, kelelahan, denyut_jantung, imt),
            'Jantung bawaan biru (Riwayat)': skor_jantung_bawaan_biru_riwayat(usia, sesak_napas, imt, riwayat_keluarga),
            'Demam reumatik (Dampak)': skor_demam_reumatik_dampak(usia, nyeri_dada, sesak_napas, kelelahan, denyut_jantung)
        }

        penyakit = pilih_penyakit_jantung_baru(all_scores, current_disease_counts, target_counts)
        
        current_disease_counts[penyakit] = current_disease_counts.get(penyakit, 0) + 1

        row = f"{nama};{usia};{jenis_kelamin};{sistolik};{diastolik};{tekanan_darah};{tekanan_cat};{kolesterol};{gula_darah};{nyeri_dada};{sesak_napas};{kelelahan};{denyut_jantung};{penyakit}"
        data.append(row)
    
    print("--- Final Generated Disease Counts (Targeted: 4 Diseases + Tidak Ada) ---")
    for disease, count in sorted(current_disease_counts.items()):
        percentage = (count / num_records) * 100
        print(f"{disease}: {count} ({percentage:.2f}%) - Target: {target_counts.get(disease, 'N/A')}")
    print("---------------------------------------------------------------------")
    return data

if __name__ == "__main__":
    dummy_data = generate_dummy_data_baru(num_records=300)
    with open("web/data_latih_2.csv", "w") as f:
        for row in dummy_data:
            f.write(row + "\n")
    print("Data dummy dengan 4 penyakit target telah disimpan ke web/data_latih_2.csv")