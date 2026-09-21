# SALES FORECASTING — INDUSTRY-GRADE SDLC DOCUMENTATION

### Technical Design, Model Validation, Deployment & Operations Handover

---

## Document Control

| Field                   | Details                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| Project ID              | SALES-DS-006                                                     |
| Project                 | Sales Forecasting                                                |
| Series Position         | Day 6 — 100-Day Industry Project Series                          |
| Domain                  | Retail / Demand Forecasting / Inventory Planning                 |
| Document Type           | SDLC + Technical Design + Model Validation + Operations Handover |
| Problem Type            | Time-Series Regression / Forecasting                             |
| Final Model             | XGBoost Regressor                                                |
| Final Feature Count     | 19                                                               |
| Dataset                 | Walmart Recruiting — Store Sales Forecasting                     |
| Source                  | Kaggle                                                           |
| Raw Records             | 421,570                                                          |
| Stores                  | 45                                                               |
| Departments             | 81                                                               |
| Historical Period       | 2010-02-05 to 2012-10-26                                         |
| Test Period             | 2012-08-10 to 2012-10-26                                         |
| Test Rows               | 35,563                                                           |
| Primary Business Metric | WMAE                                                             |
| Test WMAE               | 1,302.95                                                         |
| Naive Test WMAE         | 1,653.16                                                         |
| Improvement             | 21.18%                                                           |
| API                     | FastAPI                                                          |
| Monitoring              | PSI + WMAE performance trigger                                   |
| Document Version        | 1.0                                                              |
| Status                  | POC / Production-Architecture Demonstration                      |

---

# 1. Executive Summary

## 1.1 Business Problem

A multi-store retail organization needs reliable weekly sales forecasts at the **store-department level** to support:

* inventory replenishment,
* workforce planning,
* promotion planning,
* supply-chain coordination,
* stockout reduction,
* excess-inventory reduction,
* holiday demand preparation.

The objective of this project is to forecast the **next week's sales for each store-department combination** using historical sales behavior, calendar information, store characteristics, promotional markdowns, and regional economic variables.

The project deliberately treats forecasting differently from the earlier cross-sectional machine-learning projects in the series.

Three methodological decisions are fundamental:

1. **Chronological train/validation/test splitting** is used instead of random splitting.
2. **Naive persistence** — predicting no change from the previous week — is used as the baseline.
3. **WMAE** is the primary business metric because the original Walmart competition assigns five times the weight to holiday weeks.

---

## 1.2 Business Objective

The business objective is not simply to minimize mathematical prediction error.

The practical objective is:

> Produce sufficiently accurate weekly store-department forecasts to improve demand planning relative to an honest persistence baseline, while giving additional importance to holiday-period forecasting.

---

## 1.3 Final Outcome

The final XGBoost model achieved on the completely held-out future test period:

| Metric                |       Result |
| --------------------- | -----------: |
| Test WMAE             | **1,302.95** |
| Naive WMAE            | **1,653.16** |
| WMAE improvement      |   **21.18%** |
| Test MAE              | **1,239.03** |
| Forecasts within ±15% |   **66.60%** |
| Test observations     |   **35,563** |

The model therefore demonstrated a **21.18% reduction in WMAE relative to naive persistence** on the held-out future period.

No dollar-value inventory savings are claimed because the dataset does not contain the business's actual holding costs, stockout costs, margin impact, or replenishment economics.

---

# 2. Scope

## 2.1 In Scope

The project includes:

* raw data ingestion,
* merging of three source datasets,
* data-quality analysis,
* negative-sales treatment,
* missing-markdown handling,
* exploratory analysis,
* seasonality analysis,
* chronological data splitting,
* lag-feature generation,
* rolling-feature generation,
* model experimentation,
* naive baseline evaluation,
* Random Forest evaluation,
* XGBoost evaluation,
* WMAE-based model selection,
* held-out future evaluation,
* feature-importance analysis,
* FastAPI deployment,
* API validation,
* PSI-based drift monitoring,
* performance-decay monitoring,
* retraining-trigger simulation,
* model documentation.

## 2.2 Out of Scope

The following are not implemented or validated:

* automated inventory optimization,
* purchase-order generation,
* supply-chain optimization,
* multi-week recursive forecasting,
* real-time sales ingestion,
* production authentication,
* cloud deployment,
* enterprise CI/CD,
* automated retraining,
* inventory-dollar ROI calculation,
* stockout-cost estimation,
* holding-cost estimation,
* causal analysis of promotions,
* formal regulatory certification.

---

# 3. Stakeholder and Responsibility Model

| Stakeholder                       | Responsibility                                                 |
| --------------------------------- | -------------------------------------------------------------- |
| Demand Planning                   | Define forecasting requirements and operational use            |
| Inventory Planning                | Consume forecasts for replenishment planning                   |
| Store Operations                  | Provide business interpretation of store-level forecasts       |
| Supply Chain                      | Assess downstream inventory implications                       |
| Data Engineering                  | Productionize data ingestion and historical feature generation |
| Data Science                      | Develop, validate and maintain forecasting model               |
| ML Engineering                    | Package and deploy model                                       |
| MLOps                             | Monitor model, infrastructure and retraining workflow          |
| Business Owner                    | Approve business acceptance criteria                           |
| Model Risk / Analytics Governance | Review methodology, limitations and model-risk controls        |
| Security / IT                     | Production authentication, access control and infrastructure   |

---

# 4. SDLC Phase 1 — Business Problem Definition

## 4.1 Business Question

> Given a store, department, calendar period, store characteristics, promotion information and historical sales behavior, how accurately can next week's sales be forecast?

## 4.2 Business Decisions Supported

The forecast can support:

* inventory preparation,
* staffing decisions,
* holiday planning,
* replenishment prioritization,
* demand-planning discussions,
* exception identification.

## 4.3 Forecasting Granularity

The prediction granularity is:

**Store × Department × Week**

Each observation represents the weekly sales associated with a particular department within a particular store.

---

# 5. SDLC Phase 2 — Requirements Analysis

## 5.1 Functional Requirements

| ID    | Requirement                                                   |
| ----- | ------------------------------------------------------------- |
| FR-01 | Load train, features and stores datasets                      |
| FR-02 | Validate schema and data quality                              |
| FR-03 | Merge store, department, sales and external features          |
| FR-04 | Handle negative sales according to documented modeling policy |
| FR-05 | Handle markdown missingness                                   |
| FR-06 | Generate calendar features                                    |
| FR-07 | Generate historical sales features                            |
| FR-08 | Preserve chronological ordering                               |
| FR-09 | Establish naive forecasting baseline                          |
| FR-10 | Train candidate ML models                                     |
| FR-11 | Evaluate using WMAE                                           |
| FR-12 | Evaluate on unseen future observations                        |
| FR-13 | Expose forecasts through an API                               |
| FR-14 | Monitor feature drift                                         |
| FR-15 | Monitor forecast-performance degradation                      |
| FR-16 | Provide retraining recommendation                             |

## 5.2 Non-Functional Requirements

The system should be:

* reproducible,
* explainable,
* computationally practical,
* leakage-resistant,
* maintainable,
* deployable through a REST API,
* monitorable,
* explicit about limitations.

---

# 6. SDLC Phase 3 — Feasibility Analysis

## 6.1 Data Feasibility

The Walmart dataset contains:

* 421,570 store-department-week records,
* 45 stores,
* 81 departments,
* more than two years of historical observations,
* sales history,
* store metadata,
* economic variables,
* promotion/markdown information.

This provides sufficient historical structure for a forecasting demonstration.

## 6.2 Technical Feasibility

The project can be executed using:

* Python,
* Pandas,
* Scikit-learn,
* XGBoost,
* FastAPI,
* Joblib.

The model was intentionally trained under a **single-CPU-core compute constraint**.

## 6.3 Business Feasibility

The dataset contains the information required to demonstrate forecasting improvement against a baseline.

However, it does **not** contain the cost information required to calculate an actual inventory-dollar business case.

Therefore:

> Forecasting improvement is measured; financial ROI is not claimed.

---

# 7. SDLC Phase 4 — Project Planning

## 7.1 Development Pipeline

```text
Raw Walmart Data
       |
       v
Data Ingestion
       |
       v
Data Quality / Cleaning
       |
       v
EDA
       |
       v
Calendar + Sales-History Features
       |
       v
Chronological Split
       |
       v
Naive Baseline
       |
       v
Random Forest
       |
       v
XGBoost
       |
       v
Future Test Evaluation
       |
       v
Model Artifact
       |
       +-------------> FastAPI
       |
       +-------------> Monitoring
```

## 7.2 Planned Deliverables

* executable data pipeline,
* engineered dataset,
* EDA output,
* experiment log,
* business-validation results,
* feature-importance report,
* trained model,
* API,
* monitoring script,
* model card,
* technical documentation.

---

# 8. SDLC Phase 5 — Data Acquisition & Data Quality

## 8.1 Source Data

The project uses the real:

**Walmart Recruiting — Store Sales Forecasting** dataset.

Three source files are used:

```text
train.csv
features.csv
stores.csv
```

### train.csv

Contains:

* Store
* Dept
* Date
* Weekly_Sales
* IsHoliday

### features.csv

Contains:

* Store
* Date
* Temperature
* Fuel_Price
* MarkDown1–MarkDown5
* CPI
* Unemployment
* IsHoliday

### stores.csv

Contains:

* Store
* Type
* Size

---

## 8.2 Dataset Profile

| Attribute   |      Value |
| ----------- | ---------: |
| Rows        |    421,570 |
| Stores      |         45 |
| Departments |         81 |
| Start Date  | 2010-02-05 |
| End Date    | 2012-10-26 |

---

## 8.3 Negative Sales

The dataset contains:

**1,285 negative-sales rows**

representing approximately:

**0.305%**

of observations.

These are interpreted as weeks where returns exceeded purchases.

For this project they are **clipped to zero rather than dropped**.

### Rationale

Dropping them would remove actual store-week observations and could distort the time series.

Clipping is therefore treated as a documented modeling simplification.

### Risk

This slightly understates the variance of return-heavy store-weeks.

---

## 8.4 Markdown Missingness

The five markdown variables contain substantial missingness because the markdown program did not exist during earlier periods.

Missing markdown values are filled with zero.

Two additional features are generated:

* `TotalMarkdown`
* `HasMarkdown`

This allows the model to distinguish:

* no markdown,
* markdown activity,
* total promotional intensity.

---

# 9. SDLC Phase 6 — Exploratory Data Analysis

EDA was performed before model selection to understand:

* seasonality,
* holiday behavior,
* store differences,
* department differences,
* macroeconomic variables,
* data-quality issues.

## 9.1 Holiday Effect

| Period      | Average Weekly Sales |
| ----------- | -------------------: |
| Holiday     |          **$17,036** |
| Non-Holiday |          **$15,901** |

Holiday periods therefore exhibit materially different demand behavior.

---

## 9.2 Seasonality

The strongest observed weekly seasonal effect occurs around week 51.

| ISO Week | Average Sales |
| -------- | ------------: |
| Week 51  |   **$26,396** |
| Week 52  |   **$14,543** |

This indicates that demand peaks **before Christmas**, rather than during Christmas week itself.

This observation was measured from the dataset rather than assumed from general retail intuition.

---

## 9.3 Store-Type Analysis

| Store Type | Average Weekly Sales |
| ---------- | -------------------: |
| Type A     |          **$20,100** |
| Type B     |          **$12,237** |
| Type C     |           **$9,520** |

Store type therefore provides meaningful segmentation information.

---

## 9.4 Macro Feature Correlation

Raw correlations with weekly sales were:

| Variable     | Correlation |
| ------------ | ----------: |
| Temperature  |      -0.002 |
| Fuel Price   |     -0.0001 |
| CPI          |      -0.021 |
| Unemployment |      -0.026 |

These are effectively near-zero correlations at the raw store-week level.

This does not prove that the variables are useless.

It indicates that their **simple marginal linear relationship with sales is weak**, while temporal history and store-department identity carry considerably more signal.

---

## 9.5 Department Variance

A substantial difference exists between departments.

| Department    | Average Weekly Sales |
| ------------- | -------------------: |
| Department 92 |          **$75,205** |
| Department 43 |               **$1** |

This demonstrates that departments are not interchangeable.

`Dept` must therefore remain a first-class model feature.

---

# 10. SDLC Phase 7 — Data Preparation & Feature Engineering

## 10.1 Calendar Features

The following are extracted from `Date`:

* `Year`
* `Month`
* `WeekOfYear`

These allow the model to learn seasonal structure.

---

## 10.2 Historical Sales Features

The central feature-engineering strategy is to use each store-department's own historical sales.

### Lag 1

```text
Lag_1 = previous week's sales
```

Purpose:

* captures immediate momentum,
* provides a persistence signal,
* represents the strongest recent-sales relationship.

### Lag 52

```text
Lag_52 = sales for the same store-department approximately one year earlier
```

Purpose:

* captures year-over-year seasonal behavior,
* accounts for recurring annual demand patterns.

### Rolling 4-Week Mean

```text
Rolling_4wk_mean =
mean of previous four weeks' sales
```

The rolling window is shifted so that the current target is not included.

This prevents target leakage.

---

## 10.3 Why Time-Based Feature Engineering Is Critical

For forecasting, the model must only receive information that would genuinely be available at forecast time.

The feature pipeline therefore respects temporal ordering.

A random split would allow observations from the future to influence training.

That would produce an overly optimistic evaluation and violate the forecasting use case.

---

## 10.4 Early-Series Fallback

For store-department histories without enough observations to calculate:

* `Lag_52`,
* `Rolling_4wk_mean`,

the implementation falls back to:

1. `Lag_1`, where available;
2. otherwise the relevant series mean.

This is explicitly documented because these fallback-derived values are less reliable than genuine historical signals.

---

# 11. SDLC Phase 8 — Model Development

Three approaches were evaluated.

## 11.1 Model 1 — Naive Persistence

The baseline predicts:

```text
Next Week Sales = Last Week Sales
```

This is a deliberately simple but meaningful forecasting baseline.

A machine-learning model should demonstrate that its additional complexity provides measurable improvement over this baseline.

---

## 11.2 Model 2 — Random Forest

Configuration was deliberately reduced because the project was executed on a single CPU core.

Validation configuration:

* 15 trees
* depth 8

This is a compute-budget decision, not a claim that 15 trees is universally optimal.

---

## 11.3 Model 3 — XGBoost

XGBoost was selected as the final model after validation.

The implementation uses histogram-based tree construction.

The model contains:

**19 features**

and was trained with reduced estimator size for the available compute environment.

---

# 12. SDLC Phase 9 — Model Evaluation & Selection

## 12.1 Evaluation Metric

The primary metric is:

**Weighted Mean Absolute Error (WMAE)**

Holiday weeks receive five times the weight of non-holiday weeks.

Conceptually:

```text
WMAE =
sum(weight × absolute_error)
/
sum(weight)
```

where holiday observations receive:

```text
weight = 5
```

and ordinary observations receive:

```text
weight = 1
```

This makes the evaluation metric aligned with the original competition/business framing.

---

## 12.2 Validation Results

Validation period:

**2012-05-18 to 2012-08-03**

Duration:

**12 weeks**

| Model             | Validation WMAE | Within ±15% |
| ----------------- | --------------: | ----------: |
| Naive persistence |        1,591.01 |      64.08% |
| Random Forest     |        1,318.93 |  **67.30%** |
| XGBoost           |    **1,279.85** |      67.01% |

---

## 12.3 Model Selection Decision

Random Forest produced a slightly higher percentage of predictions within ±15%:

**67.30% vs. 67.01%**

However, XGBoost produced the lower WMAE:

**1,279.85 vs. 1,318.93**

XGBoost was therefore selected because **WMAE is the primary business metric** and explicitly assigns greater importance to holiday forecasting.

This is a documented metric tradeoff rather than a claim that XGBoost won every evaluation dimension.

---

# 13. Held-Out Future Test Evaluation

The final model was evaluated on the most recent 12 weeks:

**2012-08-10 to 2012-10-26**

Number of observations:

**35,563**

| Metric                |       Result |
| --------------------- | -----------: |
| Test WMAE             | **1,302.95** |
| Test MAE              | **1,239.03** |
| Within ±15%           |   **66.60%** |
| Naive WMAE            | **1,653.16** |
| Improvement vs. naive |   **21.18%** |

---

## 13.1 Interpretation

The final model reduced WMAE from:

**1,653.16 → 1,302.95**

representing:

**21.18% improvement**

against the naive persistence baseline.

Because the test set represents later observations in time, this is a more realistic forecasting evaluation than a random holdout.

---

# 14. Feature Importance Analysis

The top five model features were:

| Rank | Feature            | Importance |
| ---: | ------------------ | ---------: |
|    1 | `Lag_52`           | **0.5582** |
|    2 | `Lag_1`            | **0.2637** |
|    3 | `Rolling_4wk_mean` | **0.0736** |
|    4 | `IsHoliday`        | **0.0188** |
|    5 | `Type_A`           | **0.0139** |

The three sales-history features together account for:

**89.6%**

of total model decision weight.

## Business Interpretation

The strongest signal is not:

* temperature,
* fuel price,
* CPI,
* unemployment,
* store size.

Instead, the model primarily relies on the store-department's **own historical sales trajectory**.

`Lag_52` alone contributes **55.82%** importance.

This supports the EDA finding that temporal sales history is substantially more informative than raw macroeconomic correlation for this dataset.

---

# 15. SDLC Phase 10 — Deployment

## 15.1 Deployment Architecture

```text
Client / Demand Planner
        |
        | HTTP POST
        v
FastAPI
        |
        v
Input Validation
        |
        v
Feature Preparation
        |
        v
Trained XGBoost Model
        |
        v
Forecast
        |
        v
JSON Response
```

---

## 15.2 API Endpoints

| Endpoint    | Method | Purpose                       |
| ----------- | ------ | ----------------------------- |
| `/`         | GET    | Browser test form             |
| `/docs`     | GET    | Swagger/OpenAPI documentation |
| `/health`   | GET    | Service health                |
| `/forecast` | POST   | Generate forecast             |

---

## 15.3 Forecast Request

The API accepts:

* Store
* Dept
* Date
* IsHoliday
* StoreType
* StoreSize
* Lag_1
* Lag_52
* Rolling_4wk_mean

Historical sales features are explicitly supplied for the demonstration API.

In a production implementation, these should be retrieved automatically from a governed sales-history data store.

---

## 15.4 Example Forecast

A large-store, pre-Christmas holiday scenario with strong recent sales produced:

**$33,900.89**

Example response:

```json
{
  "forecast_sales": 33900.89,
  "top_factors": [
    "Lag_52",
    "Lag_1",
    "Rolling_4wk_mean"
  ]
}
```

A contrasting smaller-store, non-holiday scenario with lower recent sales produced:

**$8,806.67**

This demonstrates that the endpoint responds to different historical-sales inputs rather than returning a fixed value.

---

# 16. SDLC Phase 11 — Testing & Validation

## 16.1 Data Validation

The pipeline validates:

* source-file availability,
* required columns,
* date parsing,
* missing values,
* negative sales,
* merged dataset integrity,
* feature generation.

---

## 16.2 Leakage Validation

Particular attention is given to:

* chronological splitting,
* lag construction,
* shifted rolling windows,
* future information exclusion.

The rolling mean is shifted before being used as a predictor.

This prevents the target week's sales from entering its own features.

---

## 16.3 Model Validation

Models are evaluated using:

* WMAE,
* MAE,
* percentage within ±15%.

The primary selection criterion remains WMAE.

---

## 16.4 API Validation

The API is validated for:

* request parsing,
* required fields,
* model loading,
* forecast generation,
* response serialization,
* health endpoint functionality.

---

# 17. SDLC Phase 12 — Documentation & Handover

The handover package consists of:

```text
README.md
PROJECT_DOCUMENTATION.md
data_loader.py
eda.py
clean_and_engineer.py
split.py
train_model.py
app.py
monitor.py
requirements.txt
outputs/
```

Generated artifacts include:

```text
outputs/engineered_data.csv
outputs/eda_summary.png
outputs/experiment_log.csv
outputs/business_validation.json
outputs/feature_importance.csv
outputs/sales_forecast_model.joblib
outputs/feature_columns.joblib
outputs/model_card.json
```

---

# 18. Model Card Summary

## Intended Use

Forecast next week's store-department sales for demand-planning analysis.

## Not Intended For

The current implementation should not be treated as:

* a complete inventory optimizer,
* a multi-week forecasting engine,
* an automated purchase-order system,
* a production-ready enterprise forecasting platform.

## Model

XGBoost Regressor.

## Primary Metric

WMAE.

## Key Performance Result

21.18% WMAE improvement versus naive persistence on the held-out future test period.

## Primary Risk

The model is heavily dependent on historical sales features, particularly `Lag_52`.

Changes in store structure, department behavior, product mix or demand regime could therefore materially affect performance.

---

# 19. Explainability & Model Interpretation

The model is tree-based, so feature importance is used as a first-level explanation mechanism.

The most important features are:

1. `Lag_52`
2. `Lag_1`
3. `Rolling_4wk_mean`

The model therefore primarily uses:

* year-over-year behavior,
* immediate prior-week behavior,
* recent four-week trend.

Feature importance should not be interpreted as causal influence.

For example:

> High `Lag_52` importance means that the model relies heavily on the same store-department's prior-year sales when making predictions.

It does **not** establish that prior-year sales causally determine current sales.

---

# 20. Business Validation

## 20.1 Baseline Comparison

The central business validation is:

```text
Naive persistence WMAE = 1,653.16

Final model WMAE = 1,302.95
```

Improvement:

```text
21.18%
```

## 20.2 Forecast Accuracy

**66.60%** of test forecasts were within ±15% of actual sales.

This metric provides an additional operational interpretation but is not the primary model-selection criterion.

---

# 21. Financial Business Case

No dollar-value inventory savings are reported.

The reason is methodological:

The dataset does not provide:

* inventory holding costs,
* stockout costs,
* gross margins,
* markdown loss,
* replenishment costs,
* service-level penalties,
* working-capital costs.

Therefore, converting a 21.18% WMAE improvement into a dollar saving would require assumptions not supported by the dataset.

The project reports the measured forecasting improvement only.

A production business case should calculate:

```text
Forecast improvement
        +
Inventory economics
        +
Stockout economics
        +
Holding costs
        +
Margin impact
        =
Financial ROI
```

---

# 22. SDLC Phase 13 — Monitoring

Monitoring consists of two principal components:

1. feature-distribution drift,
2. forecasting-performance degradation.

---

## 22.1 PSI Monitoring

Population Stability Index is calculated for selected variables.

The test period is genuinely later than the beginning of the training period, making this a meaningful temporal-drift test.

Observed PSI values include:

| Feature      |       PSI | Interpretation                  |
| ------------ | --------: | ------------------------------- |
| Fuel_Price   |  **4.07** | Significant distribution change |
| CPI          |  **2.87** | Significant distribution change |
| Temperature  |  **1.31** | Significant distribution change |
| Unemployment |  **0.55** | Significant distribution change |
| Lag_1        | **0.001** | Stable                          |

---

## 22.2 Drift Interpretation

A critical monitoring principle is:

> Statistical drift does not automatically mean model failure.

### Fuel Price and CPI

These variables genuinely changed over the historical period.

Such changes may represent genuine economic regime movement.

### Temperature

The apparent drift is substantially influenced by the different seasonal composition of the test window.

Therefore, a monitoring system that blindly treats every PSI alert as a defect could generate unnecessary retraining activity.

### Lag_1

`Lag_1` remained highly stable with:

**PSI = 0.001**

This is consistent with the strong stability of recent sales behavior in the evaluated window.

---

# 23. Performance Monitoring

The monitoring process compares current WMAE against the stored baseline.

A retraining recommendation is generated when WMAE increases by more than:

**15%**

relative to the baseline.

On the current monitoring run:

**No retraining trigger was activated.**

This confirms that the monitoring logic executes successfully.

For genuine production operation, this check should be performed against newly arrived actual sales rather than repeatedly evaluating the same historical test set.

---

# 24. Production Monitoring Architecture

A production implementation should follow:

```text
New Weekly Sales
       |
       v
Data Quality Checks
       |
       v
Feature Distribution Monitoring
       |
       +----> PSI / Distribution Drift
       |
       v
Forecast vs Actual
       |
       +----> WMAE / MAE
       |
       v
Monitoring Dashboard
       |
       +----> Stable
       |
       +----> Investigate
       |
       +----> Retraining Candidate
```

Monitoring should be performed at:

* enterprise level,
* store level,
* department level,
* store-department segment level,
* holiday vs non-holiday level.

---

# 25. SDLC Phase 14 — Maintenance & Continuous Improvement

## 25.1 Recommended Retraining Triggers

Retraining should be considered when:

* WMAE deteriorates beyond the agreed threshold,
* persistent feature drift occurs,
* store composition changes,
* departments are added or removed,
* product assortment materially changes,
* promotion strategy changes,
* demand behavior changes,
* seasonal performance deteriorates.

---

## 25.2 Recommended Model Improvements

Future iterations could evaluate:

* larger Random Forest configurations,
* larger XGBoost configurations,
* LightGBM,
* CatBoost,
* dedicated time-series models,
* hierarchical forecasting,
* store-department-specific models,
* richer promotion features,
* interaction features,
* holiday-specific models,
* probabilistic forecasting,
* prediction intervals,
* ensemble forecasting.

---

# 26. Known Limitations

## Limitation 1 — Negative Sales

Negative sales are clipped to zero.

This affects approximately:

**0.305%**

of rows.

This may slightly understate variance for return-heavy store-weeks.

---

## Limitation 2 — Historical Feature Fallback

`Lag_52` and `Rolling_4wk_mean` cannot be calculated reliably for the earliest observations of a store-department series.

Fallback values therefore use `Lag_1` or the series mean.

These observations have weaker historical information.

---

## Limitation 3 — One-Week-Ahead Forecasting

The model is evaluated for one-week-ahead forecasting using the **true previous week's sales**.

For a four-week forecast:

```text
Week +1 → actual Lag_1
Week +2 → predicted Week +1
Week +3 → predicted Week +2
Week +4 → predicted Week +3
```

Prediction errors could therefore compound.

Multi-week recursive forecasting has not been evaluated.

---

## Limitation 4 — No Financial ROI

Only WMAE improvement is measured.

No inventory-dollar savings are claimed.

---

## Limitation 5 — Reduced Model Size

The Random Forest and XGBoost configurations were deliberately reduced for the single-CPU environment.

Therefore, a larger compute budget could change the relative performance of candidate models.

---

## Limitation 6 — API History Lookup

The demonstration API requires the caller to supply:

* `Lag_1`,
* `Lag_52`,
* `Rolling_4wk_mean`.

A production system should retrieve these automatically from the organization's historical sales system.

---

# 27. Productionization Gap Analysis

| Area                | Current Implementation     | Production Requirement                |
| ------------------- | -------------------------- | ------------------------------------- |
| Dataset             | Historical public dataset  | Enterprise sales/transaction data     |
| Data ingestion      | Local scripts              | Automated governed pipeline           |
| Data quality        | Script-level validation    | Production DQ framework               |
| Feature engineering | Python module              | Versioned production feature pipeline |
| Model registry      | Joblib artifact            | Model registry                        |
| Deployment          | Local FastAPI              | Containerized/cloud service           |
| Authentication      | Not implemented            | Enterprise IAM                        |
| Monitoring          | PSI + WMAE                 | Central observability platform        |
| Drift               | Historical test comparison | Continuous temporal monitoring        |
| Retraining          | Simulated trigger          | Governed automated workflow           |
| Explainability      | Feature importance         | Formal explainability framework       |
| Testing             | Script/API validation      | CI/CD automated testing               |
| Security            | POC                        | Enterprise security controls          |
| Auditability        | Local artifacts            | Central audit trail                   |
| Data lineage        | Limited                    | Enterprise lineage                    |
| Forecast horizon    | One week                   | Multi-horizon forecasting             |
| History lookup      | Caller supplied            | Automated feature retrieval           |
| Financial ROI       | Not calculated             | Actual inventory economics            |
| Governance          | Model card                 | Formal model-risk governance          |

---

# 28. Operational Runbook

## 28.1 Install

```bash
pip install -r requirements.txt
```

## 28.2 Load Data

```bash
python data_loader.py
```

## 28.3 Run EDA

```bash
python eda.py
```

Expected output includes:

```text
outputs/eda_summary.png
```

## 28.4 Engineer Features

```bash
python clean_and_engineer.py
```

## 28.5 Train and Evaluate

```bash
python train_model.py
```

Expected artifacts include:

```text
outputs/sales_forecast_model.joblib
outputs/feature_columns.joblib
outputs/experiment_log.csv
outputs/business_validation.json
outputs/feature_importance.csv
```

## 28.6 Run Monitoring

```bash
python monitor.py
```

## 28.7 Start API

```bash
uvicorn app:app --reload
```

API:

```text
/forecast
```

---

# 29. Acceptance Criteria

| Requirement                            | Status |
| -------------------------------------- | ------ |
| Real Walmart dataset used              | PASS   |
| 421,570 source rows loaded             | PASS   |
| Three source files merged              | PASS   |
| Data-quality checks performed          | PASS   |
| Negative sales handled explicitly      | PASS   |
| Markdown missingness handled           | PASS   |
| Calendar features created              | PASS   |
| Lag features created                   | PASS   |
| Rolling features created               | PASS   |
| Chronological split implemented        | PASS   |
| Naive persistence baseline implemented | PASS   |
| Random Forest evaluated                | PASS   |
| XGBoost evaluated                      | PASS   |
| WMAE used as primary metric            | PASS   |
| Future held-out test performed         | PASS   |
| 21.18% WMAE improvement demonstrated   | PASS   |
| Feature importance generated           | PASS   |
| FastAPI service implemented            | PASS   |
| Health endpoint implemented            | PASS   |
| PSI monitoring implemented             | PASS   |
| Performance monitoring implemented     | PASS   |
| Retraining trigger simulated           | PASS   |
| Known limitations documented           | PASS   |
| Productionization gaps documented      | PASS   |

---

# 30. SDLC Traceability Matrix

| SDLC Phase             | Implementation                         | Evidence / Output         |
| ---------------------- | -------------------------------------- | ------------------------- |
| 1. Business Definition | Forecast next-week sales               | Business objective        |
| 2. Requirements        | Functional/non-functional requirements | Requirement specification |
| 3. Feasibility         | Data + technical assessment            | Feasibility analysis      |
| 4. Planning            | End-to-end pipeline                    | Project structure         |
| 5. Data Acquisition    | `data_loader.py`                       | Merged dataset            |
| 6. EDA                 | `eda.py`                               | `eda_summary.png`         |
| 7. Preparation         | `clean_and_engineer.py`, `split.py`    | Engineered dataset        |
| 8. Modeling            | `train_model.py`                       | Model artifacts           |
| 9. Evaluation          | WMAE / MAE / ±15%                      | `experiment_log.csv`      |
| 10. Deployment         | `app.py`                               | FastAPI                   |
| 11. Testing            | Data/model/API validation              | Validation results        |
| 12. Documentation      | README + project documentation         | Handover package          |
| 13. Monitoring         | `monitor.py`                           | PSI + performance checks  |
| 14. Maintenance        | Retraining strategy                    | Maintenance plan          |

---

# 31. Key Technical Findings

The project produced several findings that are directly supported by execution of the real dataset.

### Finding 1 — Historical sales dominate

`Lag_52`, `Lag_1` and `Rolling_4wk_mean` collectively account for:

**89.6%**

of model feature importance.

### Finding 2 — Same-week-last-year is particularly powerful

`Lag_52` alone contributes:

**55.82%**

importance.

### Finding 3 — Raw macroeconomic correlations are weak

Temperature, fuel price, CPI and unemployment all have correlations close to zero with raw weekly sales.

### Finding 4 — Retail seasonality is not simply "Christmas week"

Week 51 averages:

**$26,396**

while week 52 averages:

**$14,543**.

### Finding 5 — Forecasting requires different validation methodology

Random splitting was deliberately avoided because future observations must never influence historical training.

### Finding 6 — Business metrics can change model selection

Random Forest had slightly better ±15% coverage, while XGBoost had lower WMAE.

Because WMAE represents the primary business objective, XGBoost was selected.

### Finding 7 — Drift requires business interpretation

Fuel Price, CPI, Unemployment and Temperature showed significant PSI values, but not every distribution change represents a model problem.

---

# 32. Final Executive Assessment

This project demonstrates a complete **time-series machine-learning SDLC**, from raw-data ingestion through model development, future-period validation, API deployment and monitoring.

The most important methodological distinction from the earlier projects in the series is that the evaluation is explicitly temporal:

```text
PAST
  |
  +---- Training
  |
  +---- Validation
  |
  +---- Future Test
```

rather than:

```text
Random observations
        |
        +---- Train
        +---- Test
```

The final XGBoost model achieved:

**1,302.95 Test WMAE**

against:

**1,653.16 Naive WMAE**

for a measured:

**21.18% improvement over persistence.**

The model's behavior is primarily driven by historical sales information, particularly:

* `Lag_52`,
* `Lag_1`,
* `Rolling_4wk_mean`.

The project does **not** claim a dollar-value inventory saving because the required inventory economics are absent from the public dataset.

The current implementation should therefore be regarded as a **production-architecture demonstration and forecasting POC**, not a production-ready enterprise demand-planning system.

Before production deployment, the organization would need:

* enterprise sales data,
* automated historical feature retrieval,
* multi-horizon evaluation,
* production-scale model tuning,
* governed data pipelines,
* authentication,
* CI/CD,
* model registry,
* continuous monitoring,
* automated retraining governance,
* inventory-cost integration,
* business acceptance testing,
* formal model governance.

---

# 33. Handover Checklist

### Data

* [x] Source documented
* [x] Schema documented
* [x] Data-quality issues documented
* [x] Negative-sales policy documented
* [x] Markdown policy documented

### Modeling

* [x] Baseline established
* [x] Candidate models evaluated
* [x] Primary metric defined
* [x] Chronological evaluation implemented
* [x] Final model documented
* [x] Feature importance documented

### Validation

* [x] Validation period documented
* [x] Future test period documented
* [x] Test WMAE documented
* [x] Baseline comparison documented
* [x] ±15% coverage documented

### Deployment

* [x] API documented
* [x] Request structure documented
* [x] Response structure documented
* [x] Health endpoint documented

### Monitoring

* [x] PSI monitoring implemented
* [x] Drift interpretation documented
* [x] Performance monitoring implemented
* [x] Retraining threshold documented

### Governance

* [x] Known limitations documented
* [x] Production gaps documented
* [x] Model-risk considerations documented
* [x] Financial-ROI limitation documented

---

# 34. Document Conclusion

**Sales Forecasting — Day 6** demonstrates an end-to-end forecasting workflow using a real retail dataset and an SDLC that explicitly addresses the unique risks of time-series machine learning.

The central engineering principle is:

> **A forecasting model must be evaluated against the future, not randomly against the past.**

The project establishes that principle through chronological validation, naive persistence benchmarking, holiday-weighted WMAE evaluation, historical-sales feature engineering, future-period testing and temporal drift monitoring.

The resulting **21.18% WMAE improvement over naive persistence** is the project's measured model-performance result.

Any further business or financial benefit must be validated using the organization's actual operational, inventory and cost data.
