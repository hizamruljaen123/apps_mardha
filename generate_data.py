import random

def generate_dummy_data(num_records=300):
    data = []
    for i in range(1, num_records + 1):
        nama = f"Pasien {i}"
        usia = random.randint(30, 70)
        jenis_kelamin = random.choice(['L', 'P'])
        tekanan_darah_systolic = random.randint(110, 170)
        tekanan_darah_diastolic = random.randint(70, 110)
        tekanan_darah = f"{tekanan_darah_systolic}/{tekanan_darah_diastolic}"
        kolesterol = random.randint(170, 280)
        gula_darah = random.randint(90, 160)
        nyeri_dada = random.choice(['Ya', 'Tidak'])
        sesak_napas = random.choice(['Ya', 'Tidak'])
        kelelahan = random.choice(['Ya', 'Tidak'])
        denyut_jantung = random.randint(60, 100)
        penyakit_jantung = random.choice([
            'Tidak Ada', 'Penyakit jantung koroner', 'Gagal jantung', 'Aritmia jantung',
            'Penyakit katup jantung', 'Kardiomiopati', 'Perikarditis', 'Endokarditis',
            'Miokarditis', 'Hipertensi pulmonal', 'Penyakit jantung bawaan',
            'Serangan jantung (infark miokard)', 'Fibrilasi atrium', 'Blok jantung',
            'Angina pektoris', 'Diseksi aorta'
        ])
        
        data.append(f"{nama};{usia};{jenis_kelamin};{tekanan_darah};{kolesterol};{gula_darah};{nyeri_dada};{sesak_napas};{kelelahan};{denyut_jantung};{penyakit_jantung}")
    return data

if __name__ == "__main__":
    dummy_data = generate_dummy_data()
    with open("web/data_latih.csv", "w") as f:  # pakai "w" untuk overwrite, bukan "a"
        for row in dummy_data:
            f.write(row + "\n")  # ganti \\n jadi \n supaya hasilnya newline asli
    print("Dummy data with 300 records generated and written to web/data_latih.csv")
