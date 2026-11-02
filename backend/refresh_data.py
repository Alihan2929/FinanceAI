
from data_fetcher import (
    BIST100_TICKERS,
    fetch_multiple_stocks,
)

from feature_engineering import (
    create_features_for_all_stocks,
)

from target_generation import (
    create_targets_for_all_stocks,
)


def refresh_bist_data():
    print("\n" + "=" * 60)
    print("FINANCEAI - BIST VERİ GÜNCELLEME")
    print("=" * 60)

    # 1. Ham verileri indir
    print("\n[1/3] HAM VERİLER GÜNCELLENİYOR...")

    successful, failed = fetch_multiple_stocks(
        BIST100_TICKERS,
        period="5d",
    )

    if not successful:
        print("Yeni veri alınamadı. İşlem durduruldu.")
        return

    # 2. Teknik özellikleri hesapla
    print("\n[2/3] TEKNİK ÖZELLİKLER HESAPLANIYOR...")

    feature_successful, feature_failed = (
        create_features_for_all_stocks(successful)
    )

    if not feature_successful:
        print("Özellik dosyaları oluşturulamadı.")
        return

    # 3. Hedef veri setlerini oluştur
    print("\n[3/3] HEDEF VERİ SETLERİ OLUŞTURULUYOR...")

    target_successful, target_failed = (
        create_targets_for_all_stocks(feature_successful)
    )

    # Rapor
    print("\n" + "=" * 60)
    print("GÜNCELLEME RAPORU")
    print("=" * 60)

    print(f"İndirilen hisse: {len(successful)}")
    print(f"İndirilemeyen hisse: {len(failed)}")

    print(f"Özellikleri hesaplanan: {len(feature_successful)}")
    print(f"Özellik hatası: {len(feature_failed)}")

    print(f"Hedefleri oluşturulan: {len(target_successful)}")
    print(f"Hedef hatası: {len(target_failed)}")

    print("\nGÜNCELLEME TAMAMLANDI.")


if __name__ == "__main__":
    refresh_bist_data()
