import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path

# Configuration
st.set_page_config(page_title="Education Decision Support", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

DATA_PATHS = [
    Path("DSS-Education_EN_20261016_113402.csv"),
    Path("../DSS-Education_EN_20261016_113402.csv"),
    Path.home() / "Downloads" / "DSS-Education_EN_20261016_113402.csv",
]

INDICATORS = [
    "Adult literacy rates",
    "Net enrolment rate in secondary education (percent)",
    "Youth literacy rates",
]

CATEGORIES = ["Nationality", "Sex", "Educational_level", "Sector"]

LABELS = {
    "Nationality": "Nationality",
    "Sex": "Sex",
    "Educational_level": "Education Level",
    "Sector": "Sector",
}

DEFAULTS = {
    "Nationality": "Total",
    "Sex": "Both sexes",
    "Educational_level": "Total",
    "Sector": "Total",
}

# Compact styling
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        max-width: 100% !important;
        padding: 1rem 1.3rem !important;
    }
    h1 {
        font-size: 1.7rem !important;
        padding-bottom: 0.2rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
        margin-top: 0.3rem !important;
    }
    div[data-testid="stMetric"] {
        padding: 0.6rem;
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    hr {
        margin: 0.5rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load and clean data
@st.cache_data
def load_data():
    for path in DATA_PATHS:
        if path.exists():
            data = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "Dataset not found. Place the CSV file in the app folder "
            "or your Downloads folder."
        )

    data = data.drop(columns=["ROW__INDEX"], errors="ignore")

    required_columns = [
        "Indicator", "Country", "Year", "Value",
        "Nationality", "Sex", "Educational_level", "Sector",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    data["Year"] = pd.to_numeric(
        data["Year"].astype(str).str.replace("Y", "", regex=False),
        errors="coerce",
    )
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna(subset=["Year", "Value"]).copy()
    data["Year"] = data["Year"].astype(int)

    for col in required_columns:
        if col not in ["Year", "Value"]:
            data[col] = data[col].astype("string").str.strip()

    return data


# Shared helpers
def options_for(data, column):
    return sorted(data[column].dropna().unique().tolist())


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


def fmt(value):
    return f"{value:,.2f}"


def make_payload(indicator, country, year, categories):
    return {
        "Indicator": indicator,
        "Country": country,
        "Year": int(year),
        **categories,
    }


def predict(payload):
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    if not result.get("success", False):
        raise ValueError(result.get("detail", "Prediction failed."))

    value = result.get("prediction", {}).get("predicted_value")

    if value is None:
        raise ValueError("No predicted value returned by the API.")

    return float(value)


def aggregate_category(data, column):
    values = options_for(data, column)

    for preferred in ["Total", "Not applicable"]:
        if preferred in values:
            return preferred

    if len(values) == 1:
        return values[0]

    return None


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# Main header and layout
st.title("Education Decision Support")
left, right = st.columns([1.05, 0.95], gap="medium")


# Section 1: Indicator selection
with left:
    st.subheader("1. Indicator Selection")

    available_indicators = [
        value for value in INDICATORS
        if value in set(df["Indicator"].dropna())
    ]

    if not available_indicators:
        st.error("None of the configured indicators exist in the dataset.")
        st.stop()

    indicator = st.selectbox(
        "Indicator",
        available_indicators,
        key="indicator",
    )

    indicator_data = df[df["Indicator"] == indicator].copy()
    years = sorted(indicator_data["Year"].unique().tolist())

    if not years:
        st.info("No years are available for this indicator.")
        st.stop()

    year = st.selectbox(
        "Year",
        years,
        index=len(years) - 1,
        key="year",
    )

    year_data = indicator_data[indicator_data["Year"] == year].copy()
    
    countries = sorted(indicator_data["Country"].dropna().unique().tolist())
    country = st.selectbox("Country", countries, key="country")
    
    # Check data availability for the selected year and country
    selected_data = indicator_data[
        (indicator_data["Year"] == year) &
        (indicator_data["Country"] == country)
        ]
    
    if selected_data.empty:
        st.warning(
            f"No data available for {country} in {year} for the selected indicator."
        )


# Section 2: Gender gap analysis
with left:
    st.divider()
    st.subheader("2. Gender Gap Analysis")

    selected_data = year_data[year_data["Country"] == country].copy()
    aggregate_filters = {}
    can_compare = True

    for col in ["Nationality", "Educational_level", "Sector"]:
        selected_value = aggregate_category(selected_data, col)

        if selected_value is None:
            can_compare = False
            break

        aggregate_filters[col] = selected_value
        selected_data = selected_data[selected_data[col] == selected_value]

    gender_data = selected_data[
        selected_data["Sex"].isin(["Male", "Female"])
    ].dropna(subset=["Value"]).copy()

    if not can_compare:
        st.info(
            "An unambiguous aggregate record is unavailable for this "
            "country, indicator, and year."
        )

    elif gender_data.empty:
        st.info(
            "No Male/Female records are available for this "
            "country, indicator, and year."
        )

    else:
        gender_summary = (
            gender_data
            .groupby("Sex", as_index=False)
            .agg(Value=("Value", "mean"))
        )

        available_sexes = set(gender_summary["Sex"])

        if not {"Male", "Female"}.issubset(available_sexes):
            st.info("Both Male and Female records are required for comparison.")

        else:
            gender_values = gender_summary.set_index("Sex")["Value"]
            male_value = float(gender_values["Male"])
            female_value = float(gender_values["Female"])
            gender_gap = male_value - female_value

            m1, m2, m3 = st.columns(3)
            m1.metric("Male", fmt(male_value))
            m2.metric("Female", fmt(female_value))
            m3.metric("Gender Gap", fmt(gender_gap))

            chart_data = pd.DataFrame({
                "Sex": ["Male", "Female"],
                "Recorded Value": [male_value, female_value],
            })

            fig = px.bar(
                chart_data,
                x="Sex",
                y="Recorded Value",
                color="Sex",
                text="Recorded Value",
                title=f"{indicator} — {country} ({year})",
                category_orders={"Sex": ["Male", "Female"]},
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                height=320,
                margin=dict(l=5, r=5, t=45, b=10),
                showlegend=False,
                xaxis_title="",
                yaxis_title="Recorded Value",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True},
            )

            st.caption(
                "Gender Gap = Male − Female. Positive values indicate a "
                "higher Male value; negative values indicate a higher "
                "Female value. Values are recorded-data averages, not "
                "causal effects."
            )


# Section 3: What-if scenario explorer
with right:
    st.subheader("3. What-if Scenario Explorer")
    st.caption("Change one category and compare model estimates.")

    st.markdown("**Baseline configuration**")
    st.write(f"**Indicator:** {indicator}")
    st.write(f"**Country:** {country} · **Year:** {year}")

    model_categories = DEFAULTS.copy()
    scenario_cols = st.columns(2)

    for i, col in enumerate(["Nationality", "Sex"]):
        values = options_for(indicator_data, col)

        if not values:
            st.error(f"No values available for {LABELS[col]}.")
            st.stop()

        with scenario_cols[i]:
            model_categories[col] = st.selectbox(
                LABELS[col],
                values,
                index=default_index(values, DEFAULTS[col]),
                key=f"model_{col}",
            )

    st.divider()

    scenario_category = st.selectbox(
        "Change category",
        ["Nationality", "Sex"],
        format_func=lambda col: LABELS[col],
        key="scenario_category",
    )

    scenario_options = options_for(indicator_data, scenario_category)
    alternative_options = [
        value for value in scenario_options
        if value != model_categories[scenario_category]
    ]

    if alternative_options:
        alternative_value = st.selectbox(
            "Alternative value",
            alternative_options,
            key="alternative_value",
        )

        compare = st.button(
            "Compare Scenarios",
            type="primary",
            use_container_width=True,
        )

        context = (
            indicator,
            country,
            year,
            tuple((col, model_categories[col]) for col in CATEGORIES),
            scenario_category,
            alternative_value,
        )

        if compare:
            baseline_payload = make_payload(
                indicator, country, year, model_categories
            )

            alternative_categories = model_categories.copy()
            alternative_categories[scenario_category] = alternative_value

            alternative_payload = make_payload(
                indicator, country, year, alternative_categories
            )

            try:
                with st.spinner("Calculating..."):
                    baseline = predict(baseline_payload)
                    alternative = predict(alternative_payload)

                difference = alternative - baseline
                percent_change = (
                    difference / abs(baseline) * 100
                    if baseline != 0 else None
                )

                st.session_state["scenario_result"] = {
                    "context": context,
                    "baseline": baseline,
                    "alternative": alternative,
                    "difference": difference,
                    "percent_change": percent_change,
                    "category": scenario_category,
                    "original": model_categories[scenario_category],
                    "alternative_category": alternative_value,
                }

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Start the API server first.")

            except requests.exceptions.Timeout:
                st.error("The API request timed out. Please try again.")

            except requests.exceptions.HTTPError as error:
                st.error(f"API error: {error}")

            except Exception as error:
                st.error(f"Could not generate estimates: {error}")

        result = st.session_state.get("scenario_result")

        if result and result["context"] == context:
            st.divider()
            st.markdown("**Scenario Results**")

            r1, r2 = st.columns(2)
            r1.metric("Baseline", fmt(result["baseline"]))
            r2.metric("Alternative", fmt(result["alternative"]))

            st.metric(
                "Difference",
                fmt(result["difference"]),
                delta=(
                    f"{result['percent_change']:.2f}%"
                    if result["percent_change"] is not None else None
                ),
            )

            scenario_chart = pd.DataFrame({
                "Scenario": ["Baseline", "Alternative"],
                "Predicted Value": [
                    result["baseline"],
                    result["alternative"],
                ],
            })

            fig = px.bar(
                scenario_chart,
                x="Scenario",
                y="Predicted Value",
                text="Predicted Value",
                title="Scenario Comparison",
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                height=300,
                margin=dict(l=5, r=5, t=45, b=10),
                showlegend=False,
                xaxis_title="",
                yaxis_title="Predicted Value",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True},
            )

            st.caption(
                f"{LABELS[result['category']]}: "
                f"{result['original']} → {result['alternative_category']}. "
                "Estimates are not observed outcomes and do not prove causation."
            )

        elif result:
            st.caption("Selections changed. Run the comparison again.")

    else:
        st.info("No alternative values are available for this category.")
