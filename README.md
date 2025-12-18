# Cart-Pole MPC Simulation

Bu proje, bir Model Predictive Control (MPC) denetleyicisi kullanarak Cart-Pole (sarkaçlı araba) sistemini dengelemeyi simüle eder.

## Gereksinimler

Bu uygulamayı çalıştırmak için Python ve aşağıdaki kütüphanelere ihtiyacınız vardır:

*   numpy
*   matplotlib
*   scipy

Kütüphaneleri yüklemek için:

```bash
pip install numpy matplotlib scipy
```

## Nasıl Çalıştırılır

Terminal veya komut satırında proje dizinine gidin ve aşağıdaki komutu çalıştırın:

```bash
python cartpole_mpc.py
```

## Çıktılar

1.  **Konsol Çıktısı**: Simülasyon sırasında zaman, açı (theta) ve konum (x) bilgileri ekrana yazdırılır.
2.  **Grafik**: Simülasyon tamamlandığında, sonuçları gösteren `cartpole_mpc_results.png` adlı bir grafik dosyası oluşturulur. Bu grafik şunları içerir:
    *   Theta (Açı) vs Zaman
    *   Konum vs Zaman
    *   Kontrol Kuvveti vs Zaman
