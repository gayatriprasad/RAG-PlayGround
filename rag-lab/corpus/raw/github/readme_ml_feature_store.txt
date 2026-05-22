# README: ML Feature Store

## Overview
The ML Feature Store provides a centralized repository for storing, versioning, and serving machine learning features used across Acme Corp's ML models. Built on Feast with custom extensions.

## Architecture
- **Feature Registry**: PostgreSQL database tracking feature definitions and metadata
- **Offline Store**: BigQuery (training data, batch features)
- **Online Store**: Redis cluster (low-latency feature serving, p99 <10ms)
- **Compute**: Apache Spark on Dataproc (feature transformations)
- **Orchestration**: Airflow DAGs for feature materialization

## Feature Categories
| Category | Features | Update Frequency | Examples |
|----------|----------|------------------|----------|
| User Profile | 23 | Daily | tenure_days, plan_tier, login_count_30d |
| Behavioral | 45 | Hourly | pages_viewed_1h, search_count_24h, last_active |
| Revenue | 18 | Daily | mrr, ltv_predicted, churn_probability |
| Content | 31 | Real-time | doc_word_count, doc_freshness, doc_view_count |

## Usage

### Python SDK
```python
from acme_features import FeatureStore

fs = FeatureStore()

# Get features for a user
features = fs.get_online_features(
    entity_keys={"user_id": "usr-123"},
    feature_names=["user_profile:tenure_days", "behavioral:login_count_30d"]
)

# Batch features for training
training_df = fs.get_historical_features(
    entity_df=entity_dataframe,
    features=["user_profile:*", "revenue:mrr"],
    start_date="2024-01-01",
    end_date="2024-06-30"
)
```

### Adding New Features
1. Define feature in `features/definitions/<category>.yaml`
2. Implement transformation in `features/transforms/<feature_name>.py`
3. Add tests in `features/tests/test_<feature_name>.py`
4. Submit PR with feature impact analysis
5. After merge, feature auto-materializes on next DAG run

## SLAs
- Online serving: p99 <10ms, 99.99% availability
- Feature freshness: within SLA of update frequency
- Offline store: queries complete within 5 minutes for 6-month windows

## Team
Owner: ML Platform Team (Slack: #ml-platform)
Feature requests: JIRA project MLPLAT
