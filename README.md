

# Geospatial Trajectory Project

Projekt analityczny i predykcyjny dla danych ruchu taxi w Porto. System łączy trajektorie kursów taxi z warstwami geograficznymi, danymi pogodowymi i infrastrukturą ulic, a następnie przewiduje poziom zatoru dla stref Porto.

Efektem projektu jest interaktywny dashboard Streamlit pokazujący na mapie PyDeck przewidywane poziomy korków w poszczególnych obszarach.

## Co robi system

- przygotowuje dane geograficzne i trajektorie taxi,
- dzieli obszar Porto na strefy H3,
- łączy kursy z infrastrukturą, pogodą i historycznymi poziomami ruchu,
- trenuje wieloklasowy model XGBoost,
- pobiera prognozę pogody Open-Meteo,
- wyświetla prognozowany poziom korków na interaktywnej mapie.

Poziomy zatoru:

| Poziom | Interpretacja |
| --- | --- |
| 0 | swobodny ruch |
| 1 | spowolnienie |
| 2 | korek |
| 3 | silny korek |

## Dashboard

Dashboard umożliwia wybór dnia i godziny oraz pokazuje:

- średni poziom ruchu we wszystkich obszarach,
- liczbę obszarów przewidywanych jako poziom 3,
- prognozę temperatury, opadów i podmuchów wiatru,
- kolorowaną mapę stref.

Mapa wykorzystuje dane statyczne obszarów oraz prognozę pogody, a przewidywanie odbywa się lokalnie przy użyciu zapisanego modelu XGBoost.

## Architektura danych

```text
dane taxi + obszary H3 + infrastruktura ulic
                    |
                    v
       przygotowanie i łączenie cech
                    |
                    v
 dane pogodowe + historyczne laghi ruchu
                    |
                    v
 model klasyfikacji XGBoost
                    |
                    v
 dashboard Streamlit + mapa PyDeck
```

Najważniejsze cechy modelu:

- rok, miesiąc, dzień, godzina i minuta,
- dzień tygodnia i pora dnia,
- identyfikator obszaru,
- opady, podmuchy wiatru, nasłonecznienie i temperatura,
- liczba pasów, skrzyżowań, sygnalizacji i innych elementów infrastruktury,
- historyczny poziom korka godzinę temu i jeden dzień temu.

## Struktura projektu

```text
data/
  processed/       przetworzone dane, laghi i modele
  data_samples/    próbki danych
  raw/             dane źródłowe
src/
  analysis/        próbkowanie i analizy
  dashboard/       aplikacja Streamlit
  data_preparation/ przygotowanie danych taxi, pogody i infrastruktury
  ml/              trening modelu
  visualisation/   pomocnicze wizualizacje
  to_shp_conversion/ eksport danych do SHP
main.py            plik startowy repozytorium
```

## Wymagania

- Python 3.10 lub nowszy,
- środowisko Windows,
- DVC z dostępem do skonfigurowanego zdalnego repozytorium danych,
- aktywne połączenie internetowe dla prognozy pogody i kafelków mapy.

## Instalacja

Utwórz i aktywuj wirtualne środowisko, następnie zainstaluj zależności:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pobierz dane zarządzane przez DVC:

```powershell
dvc pull
```

Jeśli środowisko ma skonfigurowany inny remote niż projektu, pobranie danych może wymagać wcześniejszej konfiguracji credentials DVC.

## Uruchomienie dashboardu

Z katalogu głównego repozytorium uruchom:

```powershell
python -m streamlit run src/dashboard/app.py
```

Następnie otwórz adres wyświetlony przez Streamlit, domyślnie:

```text
http://127.0.0.1:8501
```

## Trenowanie modelu

Do ponownego trenowania modelu użyj:

```powershell
python src/ml/training.py
```

Skrypt:

- łączy dane taxi, pogodowe i infrastrukturalne,
- tworzy poziomy korków,
- dodaje cechy lagowe,
- dzieli dane chronologicznie,
- trenuje klasyfikator XGBoost,
- zapisuje model jako `data/processed/xgboost_traffic_model_heatmap.pkl`.

## Przygotowanie danych

Poszczególne moduły można uruchamiać z katalogu głównego repozytorium. Dane przetwarzane powinny być zapisane zgodnie ze ścieżkami oczekiwanymi przez dashboard i skrypt treningowy.

Najważniejsze artefakty:

- `data/processed/area.parquet`,
- `data/processed/traffic.parquet`,
- `data/processed/weather.parquet`,
- `data/processed/taxi_prepared_full.parquet`,
- `data/processed/lag_history.parquet`,
- `data/processed/xgboost_traffic_model_heatmap.pkl`.

Duże dane i modele są śledzone przez DVC, dlatego zwykle nie powinny być przechowywane bezpośrednio w Git.

## Przepływ pracy

Po zmianach w danych lub kodzie przygotowania:

```powershell
dvc add data/data_samples
dvc add data/processed/area.parquet
dvc add data/processed/traffic.parquet
dvc push
git add .
git commit -m "Update processed traffic data"
git push
```

Polecenia powinny być dostosowane do faktycznych plików zmodyfikowanych przez konkretny etap projektu.

## Ograniczenia

- Prognoza pogody jest pobierana z Open-Meteo i może zależeć od dostępności zewnętrznego API.
- Model przewiduje poziom ruchu dla stref statystycznych, a nie indywidualną trasę pojazdu.
- Jakość predykcji dla obszarów zależy od reprezentacji danych treningowych.
- Wyniki historyczne mogą być ograniczone dla obszarów, w których zebrano niewiele obserwacji.

## Autorzy projektu

- **Kacper Jedynak** - inicjalizacja repozytorium i konfiguracja DVC, przygotowanie próbek danych, pobieranie i przekształcanie danych pogodowych, implementacja workflow ML oraz trenowanie modeli XGBoost.
- **Gabriela Wełna** - przygotowanie danych infrastruktury i ruchu ulicnego, prace z danymi GeoJSON oraz integracja danych pogodowych przez API.
- **Szymon Jurczak** - przygotowanie danych taxi i obszarów H3, dobór próbek oraz prace nad przygotowaniem danych wejściowych.
- **Kamil Kucia** - generowanie danych CSV i ich przekształcenie do formatu Parquet, przygotowanie danych pogodowych oraz implementacja dashboardu Streamlit ze historycznymi lagami ruchu.
- **Jan Serzysko** - konwersja danych geograficznych do formatu SHP i GeoJSON, przygotowanie danych do dalszej analizy oraz prace nad wizualizacją projektu.

