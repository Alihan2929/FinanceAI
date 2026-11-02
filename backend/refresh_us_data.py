
from us_tickers import US_TICKERS

from data_fetcher import fetch_multiple_stocks

from feature_engineering import (
    create_features_for_all_stocks,
)

from target_generation import (
    create_targets_for_all_stocks,
)


def refresh_us_data():
    print("\n" + "=" * 60)
    print("FINANCEAI - ABD HISSE VERİLERİ")
    print("=" * 60)

    # 1. Ham verileri indir
    print("\n[1/3] ABD HİSSE VERİLERİ İNDİRİLİYOR...")

    successful, failed = fetch_multiple_stocks(
        US_TICKERS,
        period="5y",
    )

    if not successful:
        print("Hiçbir ABD hissesi indirilemedi.")
        return

    # 2. Teknik özellikleri hesapla
    print("\n[2/3] TEKNİK ÖZELLİKLER HESAPLANIYOR...")

    feature_successful, feature_failed = (
        create_features_for_all_stocks(successful)
    )

    # 3. Hedef veri setlerini oluştur
    print("\n[3/3] HEDEF VERİ SETLERİ OLUŞTURULUYOR...")

    target_successful, target_failed = (
        create_targets_for_all_stocks(feature_successful)
    )

    print("\n" + "=" * 60)
    print("ABD VERİ GÜNCELLEME RAPORU")
    print("=" * 60)

    print(f"İndirilen: {len(successful)}")
    print(f"İndirme hatası: {len(failed)}")
    print(f"Özellik hesaplanan: {len(feature_successful)}")
    print(f"Özellik hatası: {len(feature_failed)}")
    print(f"Hedef oluşturulan: {len(target_successful)}")
    print(f"Hedef hatası: {len(target_failed)}")

    print("\nABD VERİ GÜNCELLEMESİ TAMAMLANDI.")


if __name__ == "__main__":
    refresh_us_data()
