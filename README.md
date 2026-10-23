# DSS_Education Decision Support System
An education data analytics and machine learning decision support system for exploring educational indicators across Arab countries

## Project Overview

Education statistics can reveal important patterns in literacy, school enrolment, and gender disparities. However, interpreting these indicators across countries and years requires more than simply viewing raw data tables.

This project transforms publicly available education statistics into an interactive decision support application. Users can explore selected indicators, examine historical data, compare male and female outcomes, and test prediction scenarios using a trained machine learning model.

## Data Source

The project uses the **DSS-Education EN** dataset, listed in the [Arab Development Portal Data Catalog](https://unrcp-p-webapp-001-adp-staging.azurewebsites.net/datacatalog/un-agencies/ccd3f7cd-a6e2-5ded-ad3c-4cbc3c0f5534).

- **Publisher / Data source:** United Nations Economic and Social Commission for Western Asia (UN ESCWA), as identified by the portal.
- **Dataset:** Demographic and Social Statistics — Education.
- **Geographic coverage:** Arab countries.
- **Time coverage:** The portal describes coverage from 2001 to 2023; actual availability varies by indicator and country.
- **Dimensions:** Indicator, country, year, nationality, sex, educational level, and sector.
- **Target variable:** `Value`.

The dataset includes education indicators such as adult literacy, youth literacy, school enrolment, and public expenditure on education.

## Project Objectives

- Explore education indicators across countries and years.
- Identify historical patterns and differences in educational outcomes.
- Compare male and female indicator values where paired observations are available.
- Provide an interactive interface for exploring selected indicators.
- Demonstrate machine learning predictions through a FastAPI service.

## Application Features

### 1. Education Indicator Explorer
Select an indicator, year, and country to explore available observations.

### 2. Gender Comparison
Compare recorded male and female values and calculate the gender gap when both observations are available.

### 3. What-if Scenario Explorer
Change supported model input categories and compare the model's baseline prediction with an alternative scenario.

## Machine Learning

The project includes a trained Random Forest model and additional model evaluation artifacts. Predictions are served through a FastAPI backend and displayed in a Streamlit application.
Predictions are model-based estimates, not official statistics or proof of causal relationships. Their reliability depends on data coverage, preprocessing, and model evaluation.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- FastAPI
- Plotly
- Joblib
- Jupyter Notebook

## Project Structure

- `app.py` — Interactive Streamlit application.
- `api/` — FastAPI prediction service.
- `modeling/` — Trained models and evaluation artifacts.
- `plots/` — Data visualization outputs.
- `reports/` — Analytical summaries and comparisons.
- `notebook.ipynb` — Exploratory analysis and model development.
- `requirements.txt` — Python dependencies.
- `DSS-Education_EN_20261016_113402.csv` — Local dataset file used by the project.

## Limitations

- Data availability differs across indicators, countries, and years.
- Missing observations should not be interpreted as zero.
- Gender comparisons require valid observations for both groups.
- Model predictions should be interpreted alongside model evaluation results and the limitations of the available data.

## Acknowledgment

Data source: UN ESCWA, as listed in the Arab Development Portal Data Catalog.

This project is an independent educational and analytical project and is not an official product of UN ESCWA or the Arab Development Portal.
