import random
from test_data import generate_dummy_data, hitung_imt, kategori_tekanan_darah


if __name__ == "__main__":
    dummy_data = generate_dummy_data()
    with open("web/data_latih_2.csv", "w") as f:
        for row in dummy_data:
            f.write(row + "\n")
    print("Data dummy realistis dengan kolom yang diminta telah disimpan ke web/data_latih_2.csv")
