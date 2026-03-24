# Norgespris og belastning i strømnettet i Agder

## Om prosjektet
Dette prosjektet undersøker i hvilken grad **Norgespris** har påvirket belastningsnivået i strømnettet i Agder i perioder med begrenset nettkapasitet. Prosjektet kombinerer datainnhenting, databehandling, feature engineering, statistisk analyse og visualisering for å utforske om endrede prismekanismer kan ha påvirket belastning i utvalgte høylastperioder.

Analysen er gjennomført i Python, med **DuckDB** for datalagring og **Streamlit** for visualisering av resultater.

## Problemstilling
**I hvilken grad har Norgespris påvirket belastningsnivået i strømnettet i Agder i perioder med begrenset nettkapasitet?**

## Formål
Formålet med prosjektet er å analysere om innføringen av Norgespris har ført til endringer i strømforbruksmønstre som kan bidra til økt belastning i strømnettet i Agder, særlig i perioder med begrenset kapasitet.

## Prosjektmål
- hente inn og bearbeide relevante data
- klargjøre datasett for analyse gjennom feature engineering
- gjennomføre regresjonsanalyse og annen statistisk analyse
- visualisere resultater i et interaktivt dashboard
- utforske sammenhenger mellom prisordning, forbruksmønster og belastning i strømnettet

## Teknologier og verktøy
Prosjektet benytter blant annet:

- **Python**
- **Pandas**, **NumPy** og **SciPy**
- **Statsmodels** og **Scikit-learn**
- **Matplotlib**
- **Requests**
- **Azure Blob Storage**
- **python-dotenv**
- **DuckDB**
- **Streamlit**
- **PySpark**
- **holidays** og **arrow**

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
│   │   └── spark_utils.py        # Hjelpefunksjoner for Spark
│   │
│   ├── assets/images/            # Logoer og andre bildefiler
│   │
│   ├── dashboard/
│   │   └── app.py                # Streamlit-applikasjon
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── duckdb_utils.py       # Hjelpefunksjoner for DuckDB
│   │
│   └── feature_engineering/
│       ├── __init__.py
│       ├── forbruksdata.py       # Behandling av forbruksdata
│       └── værdata.py            # Behandling av værdata
│
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
````

## Installasjon

### 1. Klon prosjektet

```bash
git clone <repo-url>
cd <prosjektnavn>
```

### 2. Opprett virtuelt miljø

På macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

På Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installer avhengigheter

```bash
pip install -r requirements.txt
```

## Kjøring

### Start Streamlit-appen

```bash
streamlit run src/dashboard/app.py
```

### Jobb med notebooks

```bash
jupyter notebook
```

## Avhengigheter

Prosjektet bruker følgende Python-pakker:

```txt
pandas
numpy
scipy
matplotlib
statsmodels
scikit-learn
geopy
requests
azure-storage-blob
azure-identity
python-dotenv
duckdb
streamlit
pyspark
holidays
arrow
```

## Data og behandling

Prosjektet bygger på datasett fra trafostasjonene:

- Breive
- Frikstad
- Hartevatn
- Timenes

For hver stasjon benyttes blant annet:
- forbruksdata
- værdata
- antall brukere med Norgespris
- tidsvariabler og kalenderdata
- informasjon om perioder med begrenset nettkapasitet

Data hentes inn, vaskes og transformeres før de brukes i analyse og visualisering. Værdata hentes fra Frost API, som er Meteorologisk institutts API for historiske vær- og klimadata. Deler av databehandlingen er organisert i egne moduler for forbruksdata, værdata og databasehåndtering.

Dataene lagres i Azure Blob Storage. Tilgang til lagringen krever autentisering gjennom hemmelige nøkler eller miljøvariabler som ikke er versjonstyrt i GitHub av sikkerhetshensyn. For å kjøre prosjektet med direkte datatilgang må gyldige miljøvariabler derfor settes opp lokalt, for eksempel i en `.env`-fil som ikke deles i repositoryet.

Dersom slik tilgang ikke er satt opp, må data lastes ned og lagres lokalt før analyse og kjøring av prosjektet.

## Analyse

Analysen fokuserer på om Norgespris kan ha påvirket belastningsnivået i strømnettet i Agder. Dette undersøkes ved hjelp av statistiske metoder og regresjonsanalyse, samt utforskende analyser i notebooks.

## Dashboard

Prosjektet inkluderer et dashboard bygget i Streamlit for å presentere resultater og visualiseringer på en mer tilgjengelig måte.

## Videre arbeid

Mulige videreutviklinger av prosjektet er:

* teste flere modeller og forklaringsvariabler
* utvide analysen til flere geografiske områder
* forbedre dashboard og visualiseringer
* videreutvikle datarørledningene og databaseoppsettet

## Merknader

Mapper og filer som `__pycache__` og `.DS_Store` bør normalt ikke versjonstyres og bør legges i `.gitignore`.

## Lisens

Dette prosjektet er utviklet som en del av et bachelorprosjekt og er ikke ment for kommersiell bruk.

````
