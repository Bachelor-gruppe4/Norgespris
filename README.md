# Norgespris og belastning i strømnettet i Agder

## Om prosjektet
Dette prosjektet undersøker i hvilken grad **Norgespris** har påvirket belastningsnivået i strømnettet i Agder i perioder med begrenset nettkapasitet. Prosjektet kombinerer datainnhenting, databehandling, feature engineering, statistisk analyse og visualisering for å utforske om endrede prismekanismer kan ha påvirket belastning i utvalgte høylastperioder.

Analysen er gjennomført i Python, med **DuckDB** for datalagring og **Streamlit** for visualisering.

## Problemstilling
**I hvilken grad har Norgespris påvirket belastningsnivået i strømnettet i Agder i perioder med begrenset nettkapasitet?**

## Formål
Formålet er å finne ut om innføringen av Norgespris har ført til endringer i strømforbruksmønstre som kan bidra til økt belastning i strømnettet i Agder i kritiske perioder.

## Prosjektmål
- Hente inn og bearbeide relevante data
- Klargjøre datasett for analyse gjennom feature engineering
- Gjennomføre regresjonsanalyse og annen statistisk analyse
- Visualisere resultater i et interaktivt dashboard
- Utforske sammenhenger mellom prisordning, forbruksmønster og belastning i strømnettet

## Teknologier og verktøy
Prosjektet benytter blant annet:

- **Python**
- **Pandas** og **NumPy**
- **Linearmodels**
- **Matplotlib**
- **Altair**
- **DuckDB**
- **Jupyter**
- **ipykernel**
- **Streamlit**
- **Requests**
- **geopy**
- **Azure-Storage-Blob**
- **python-dotenv**
- **holidays**

## Prosjektstruktur
```bash
.
├── notebooks/
│   ├── analysis/                 # Notebooks for analyse
│   ├── exploration/              # Utforskende analyser
│   └── feature_engineering/      # Klargjøring og bearbeiding av data
│
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── regression.py         # Regresjonsanalyse
│   │   └── weather_regression.py # Værmodell for panelregresjon
│   │
│   ├── dashboard/
│   │   ├── app.py                # Streamlit-applikasjonen
│   │   ├── config.py             # App-konfigurasjon og tema
│   │   ├── queries.py            # Datainnhenting og spørringer
│   │   └── styles.py             # Tilpasning av utseende i dashboardet
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── duckdb_utils.py       # DuckDB / Azure Blob Storage-tilgang
│   │
│   └── feature_engineering/
│       ├── __init__.py
│       ├── forbruksdata.py       # Forbruksdata rensing og feature engineering
│       ├── norgespris.py         # Norgespris-data rensing og stasjonsdeling
│       └── værdata.py            # Vær API-henting og Azure-opplast
│
├── .streamlit/
│   └── config.toml               # Streamlit-tema og app-konfigurasjon
├── .gitignore
├── environment.yml
└── README.md
```

## Feature engineering

Prosjektet inkluderer egne moduler for dataklargjøring og feature engineering. Dette omfatter blant annet generering av tidsvariabler, håndtering av Norgespris-perioder og kobling av værdata til forbruksdata på timebasis.

## Installasjon

### 0. Installer Conda (Miniconda eller Anaconda)
Conda må være installert for å kunne sette opp miljøet.

Last ned her:
https://www.anaconda.com/download/success

Etter installasjon, lukk og åpne terminalen på nytt og sjekk at det fungerer:
```bash
conda --version
```

### 1. Klon prosjektet

```bash
git clone <repo-url>
cd <prosjektnavn>
```

### 2. Opprett og aktiver Conda-miljø

```bash
conda env create -f environment.yml
conda activate bachelor2026
```

### 3. Installer Jupyter-kernel

```bash
python -m ipykernel install --user --name bachelor2026 --display-name "Bachelor 2026"
```

### 4. Konfigurer miljøvariabler

Opprett en `.env`-fil i prosjektets rotkatalog med følgende variabler:

```env
AZURE_STORAGE_CONNECTION_STRING="<din-azure-connection-string>"
FROST_CLIENT_ID="<din-frost-client-id>"
```

Disse lastes automatisk ved hjelp av `python-dotenv` og brukes av både databehandling, analyse og visualisering.

Hvis du kjører appen i Streamlit Cloud eller en annen skyplattform, kan variablene legges inn som sikre `secrets` eller miljøvariabler i plattformen.

## Kjøring

### Start Streamlit-appen

```bash
streamlit run src/dashboard/app.py
```

### Jobb med notebooks

```bash
jupyter notebook
```

## Data og behandling

Prosjektet benytter data fra trafostasjonene:

- Breive
- Frikstad
- Hartevatn
- Timenes

For hver stasjon brukes blant annet:
- Forbruksdata
- Værdata
- Antall brukere med Norgespris
- Tidsvariabler og kalenderdata
- Informasjon om perioder med begrenset nettkapasitet

Data hentes inn, vaskes og transformeres før analyse og visualisering. Værdata hentes fra Frost API, Meteorologisk institutt sitt API for historiske vær- og klimadata. Datahåndteringen er organisert i egne moduler for forbruksdata, værdata og database.

Data lagres i Azure Blob Storage. Tilgang krever autentisering med hemmelige nøkler eller miljøvariabler som ikke er versjonstyrt i GitHub. For direkte datatilgang må gyldige miljøvariabler være satt opp lokalt.

Hvis slik tilgang ikke er tilgjengelig, må data lastes ned og lagres lokalt før prosjektet kan kjøres fullt ut.

## Analyse

Analysen fokuserer på om Norgespris kan ha påvirket belastningsnivået i strømnettet i Agder. Dette undersøkes med statistiske metoder, regresjonsanalyse og utforskende notebooks.

## Dashboard

Streamlit-dashboardet er bygget i `src/dashboard/`. App-koden ligger i `src/dashboard/app.py`, mens `src/dashboard/config.py`, `src/dashboard/queries.py` og `src/dashboard/styles.py` håndterer konfigurasjon, datainnhenting og utseende.

## Videre arbeid

Mulige videreutviklinger er:

- Teste flere modeller og forklaringsvariabler
- Skalere analysen til flere nivåer i strømnettet
- Utvide analysen til flere geografiske områder
- Forbedre dashboard og visualiseringer
- Videreutvikle datarørledningene og databaseoppsettet

## Merknader

Mapper og filer som `__pycache__` og `.DS_Store` bør normalt ikke versjonstyres og bør legges i `.gitignore`.

## Lisens

Dette prosjektet er utviklet som en del av et bachelorprosjekt og er ikke ment for kommersiell bruk.
