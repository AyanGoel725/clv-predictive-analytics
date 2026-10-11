# Grouped out-of-fold churn evaluation

Run: f0095e2660d447bd9df382f307062afb | Model: v2.1.0-grouped-sigmoid-unweighted
Prediction source: grouped OOF | Labeled arrays: 1884 | Accounts: 995
Observed 12-month array-event rate: 15.8174%

Outer folds keep accounts intact. Capacity bins fit within each training fold. Calibrators fit on inner account-grouped OOF scores; outer labels never fit the mapping.
This is cross-sectional validation, not historical back-testing. Observation dates and label maturity are unverified.

## Probability metrics

Average precision is reported as AP, not trapezoidal PR area. Lower Brier/log loss is better. Positive bias means over-forecasting churn. Candidates and fixed weights were specified before this run; no outer-fold threshold tuning.

| model                        |   mean_probability |   observed_rate |      bias |    brier |   log_loss |   roc_auc |   average_precision |   ece_equal_width |   ece_equal_count |   brier_lift_vs_5_pct |
|:-----------------------------|-------------------:|----------------:|----------:|---------:|-----------:|----------:|--------------------:|------------------:|------------------:|----------------------:|
| static_5                     |           0.050000 |        0.158174 | -0.108174 | 0.144857 |   0.517027 |  0.500000 |            0.158174 |          0.108174 |          0.108174 |              0.000000 |
| training_prevalence          |           0.158173 |        0.158174 | -0.000001 | 0.133165 |   0.436668 |  0.491357 |            0.155553 |          0.000001 |          0.004959 |              8.071123 |
| logistic                     |           0.158117 |        0.158174 | -0.000057 | 0.133386 |   0.437141 |  0.528420 |            0.171996 |          0.011674 |          0.029564 |              7.918510 |
| weighted_xgb_raw             |           0.392007 |        0.158174 |  0.233833 | 0.206507 |   0.600878 |  0.523693 |            0.167561 |          0.235432 |          0.235247 |            -42.559381 |
| weighted_xgb_sigmoid         |           0.157854 |        0.158174 | -0.000320 | 0.133165 |   0.436666 |  0.500421 |            0.160228 |          0.000320 |          0.025582 |              8.070889 |
| weighted_xgb_isotonic        |           0.160505 |        0.158174 |  0.002331 | 0.133234 |   0.436866 |  0.511063 |            0.161111 |          0.002331 |          0.024166 |              8.023488 |
| weighted_rf_raw              |           0.439119 |        0.158174 |  0.280945 | 0.218297 |   0.628090 |  0.510410 |            0.169290 |          0.280945 |          0.280945 |            -50.698587 |
| weighted_rf_sigmoid          |           0.157956 |        0.158174 | -0.000218 | 0.133220 |   0.436888 |  0.491659 |            0.163200 |          0.000218 |          0.013073 |              8.033193 |
| weighted_rf_isotonic         |           0.159335 |        0.158174 |  0.001161 | 0.133093 |   0.436272 |  0.517504 |            0.166091 |          0.001305 |          0.009629 |              8.120618 |
| weighted_ensemble_raw        |           0.410852 |        0.158174 |  0.252678 | 0.208659 |   0.606657 |  0.521567 |            0.167605 |          0.252678 |          0.252678 |            -44.045346 |
| weighted_ensemble_sigmoid    |           0.157902 |        0.158174 | -0.000273 | 0.133156 |   0.436635 |  0.496132 |            0.163919 |          0.000273 |          0.018886 |              8.077494 |
| weighted_ensemble_isotonic   |           0.160171 |        0.158174 |  0.001997 | 0.133015 |   0.436012 |  0.525878 |            0.170384 |          0.001997 |          0.007410 |              8.175106 |
| unweighted_xgb_raw           |           0.154005 |        0.158174 | -0.004169 | 0.138421 |   0.456958 |  0.526583 |            0.172597 |          0.047526 |          0.054734 |              4.442721 |
| unweighted_xgb_sigmoid       |           0.158121 |        0.158174 | -0.000053 | 0.133153 |   0.436635 |  0.507274 |            0.170930 |          0.000053 |          0.013142 |              8.079445 |
| unweighted_xgb_isotonic      |           0.159626 |        0.158174 |  0.001452 | 0.133321 |   0.436867 |  0.526230 |            0.171240 |          0.001830 |          0.015318 |              7.963247 |
| unweighted_rf_raw            |           0.158419 |        0.158174 |  0.000245 | 0.134568 |   0.441621 |  0.523721 |            0.171433 |          0.024571 |          0.031569 |              7.102708 |
| unweighted_rf_sigmoid        |           0.157661 |        0.158174 | -0.000513 | 0.133376 |   0.437471 |  0.488134 |            0.155305 |          0.000513 |          0.021068 |              7.925670 |
| unweighted_rf_isotonic       |           0.158503 |        0.158174 |  0.000329 | 0.133268 |   0.437046 |  0.512087 |            0.163147 |          0.003448 |          0.014421 |              8.000243 |
| unweighted_ensemble_raw      |           0.155771 |        0.158174 | -0.002403 | 0.136109 |   0.447473 |  0.527317 |            0.174157 |          0.037904 |          0.040547 |              6.038821 |
| unweighted_ensemble_sigmoid  |           0.157952 |        0.158174 | -0.000222 | 0.133202 |   0.436824 |  0.500004 |            0.166414 |          0.000222 |          0.019726 |              8.045411 |
| unweighted_ensemble_isotonic |           0.159334 |        0.158174 |  0.001160 | 0.133355 |   0.437316 |  0.510843 |            0.159539 |          0.004404 |          0.017256 |              7.939883 |

Prespecified candidate: unweighted_ensemble_sigmoid.
Brier score improves by 8.05% relative to the baseline.
No reliable Brier benefit over a fold-trained portfolio rate is established. Retention targeting and a production champion claim are unsupported.

## Paired account bootstrap intervals

1000 resamples; seed 20261011. Positive gain = lower Brier. Intervals condition on predictions and exclude refitting/model-selection uncertainty.

| model                       | baseline            |   brier_gain |   gain_ci_low |   gain_ci_high |   bias_ci_low |   bias_ci_high |
|:----------------------------|:--------------------|-------------:|--------------:|---------------:|--------------:|---------------:|
| weighted_ensemble_raw       | static_5            |    -0.063803 |     -0.076486 |      -0.050583 |      0.234993 |       0.269624 |
| weighted_ensemble_raw       | training_prevalence |    -0.075494 |     -0.084967 |      -0.065752 |      0.234993 |       0.269624 |
| unweighted_xgb_raw          | static_5            |     0.006436 |      0.001600 |       0.010859 |     -0.021511 |       0.012113 |
| unweighted_xgb_raw          | training_prevalence |    -0.005256 |     -0.008145 |      -0.002559 |     -0.021511 |       0.012113 |
| unweighted_xgb_sigmoid      | static_5            |     0.011704 |      0.008146 |       0.015451 |     -0.017162 |       0.016188 |
| unweighted_xgb_sigmoid      | training_prevalence |     0.000012 |     -0.000193 |       0.000213 |     -0.017162 |       0.016188 |
| unweighted_rf_raw           | static_5            |     0.010289 |      0.006020 |       0.014525 |     -0.016709 |       0.016222 |
| unweighted_rf_raw           | training_prevalence |    -0.001403 |     -0.003098 |       0.000241 |     -0.016709 |       0.016222 |
| unweighted_rf_sigmoid       | static_5            |     0.011481 |      0.007973 |       0.015190 |     -0.017584 |       0.015794 |
| unweighted_rf_sigmoid       | training_prevalence |    -0.000211 |     -0.000515 |       0.000090 |     -0.017584 |       0.015794 |
| unweighted_ensemble_raw     | static_5            |     0.008748 |      0.004163 |       0.012820 |     -0.019561 |       0.013614 |
| unweighted_ensemble_raw     | training_prevalence |    -0.002944 |     -0.005208 |      -0.000773 |     -0.019561 |       0.013614 |
| unweighted_ensemble_sigmoid | static_5            |     0.011654 |      0.008085 |       0.015435 |     -0.017390 |       0.016041 |
| unweighted_ensemble_sigmoid | training_prevalence |    -0.000037 |     -0.000289 |       0.000220 |     -0.017390 |       0.016041 |
| training_prevalence         | static_5            |     0.011692 |      0.008168 |       0.015355 |     -0.016956 |       0.016252 |
| training_prevalence         | training_prevalence |     0.000000 |      0.000000 |       0.000000 |     -0.016956 |       0.016252 |
| logistic                    | static_5            |     0.011470 |      0.007724 |       0.015334 |     -0.017345 |       0.015986 |
| logistic                    | training_prevalence |    -0.000221 |     -0.001358 |       0.000789 |     -0.017345 |       0.015986 |

## Diagnostic cohorts

Contract expiry year and expansion tags are proxy segments, not product families or historical windows. Cohort variability does not prove feature usefulness.

| dimension                  | cohort   |   n |   mean_probability |   observed_rate |      bias |    brier |   log_loss |   roc_auc |   average_precision |   ece_equal_width |   ece_equal_count |   brier_gain |   gain_ci_low |   gain_ci_high |   bias_ci_low |   bias_ci_high |
|:---------------------------|:---------|----:|-------------------:|----------------:|----------:|---------:|-----------:|----------:|--------------------:|------------------:|------------------:|-------------:|--------------:|---------------:|--------------:|---------------:|
| contract_year              | 2026     | 171 |           0.159158 |        0.169591 | -0.010432 | 0.140912 |   0.455555 |  0.533269 |            0.200478 |          0.010432 |          0.075908 |     0.014220 |      0.002386 |       0.027025 |     -0.068968 |       0.042095 |
| contract_year              | 2027     | 684 |           0.157844 |        0.152047 |  0.005797 | 0.128658 |   0.425257 |  0.535693 |            0.196039 |          0.005797 |          0.034022 |     0.010684 |      0.004350 |       0.016641 |     -0.021730 |       0.035297 |
| contract_year              | 2028     | 697 |           0.158019 |        0.159254 | -0.001235 | 0.134170 |   0.439511 |  0.477939 |            0.158537 |          0.001235 |          0.037943 |     0.011659 |      0.006119 |       0.017810 |     -0.029842 |       0.024431 |
| contract_year              | 2029     | 332 |           0.157414 |        0.162651 | -0.005237 | 0.136564 |   0.445368 |  0.463096 |            0.147210 |          0.005237 |          0.058236 |     0.012322 |      0.004330 |       0.020983 |     -0.045180 |       0.031620 |
| expansion_potential_signal | High     | 401 |           0.157515 |        0.144638 |  0.012877 | 0.123644 |   0.413010 |  0.540213 |            0.178447 |          0.012877 |          0.029269 |     0.009030 |      0.001328 |       0.016248 |     -0.020961 |       0.049286 |
| expansion_potential_signal | Low      | 790 |           0.157239 |        0.151899 |  0.005340 | 0.128789 |   0.425887 |  0.503470 |            0.179697 |          0.005340 |          0.044794 |     0.010420 |      0.005239 |       0.016207 |     -0.021859 |       0.029292 |
| expansion_potential_signal | Medium   | 588 |           0.159130 |        0.180272 | -0.021142 | 0.148587 |   0.474811 |  0.470348 |            0.176424 |          0.021142 |          0.050570 |     0.016158 |      0.009159 |       0.022839 |     -0.052329 |       0.010956 |
| expansion_potential_signal | UNKNOWN  | 105 |           0.158388 |        0.133333 |  0.025054 | 0.116753 |   0.397343 |  0.454474 |            0.135284 |          0.025054 |          0.071747 |     0.005747 |     -0.007702 |       0.019749 |     -0.040455 |       0.089596 |

## Signed historical exposure by OOF risk bucket

Policy thresholds: CRITICAL >=0.70; HIGH >=0.40; MEDIUM >=0.15; LOW <0.15. Buckets do not establish retention uplift. Empty-bucket observed rates are undefined.

| risk_bucket   |      arrays |     churns |   observed_rate |   signed_historical_profit_usd |
|:--------------|------------:|-----------:|----------------:|-------------------------------:|
| CRITICAL      |    0.000000 |   0.000000 |      nan        |                       0.000000 |
| HIGH          |    0.000000 |   0.000000 |      nan        |                       0.000000 |
| MEDIUM        | 1661.000000 | 263.000000 |        0.158338 |              3543978757.000000 |
| LOW           |  223.000000 |  35.000000 |        0.156951 |               509152868.000000 |

Signed labeled historical profit total: $4,053,131,625.00. This is neither expected future loss nor discounted CLV.

## Limitations and recommendation

Array labels do not establish account cancellation. Future expansion outcomes, product families, observation timestamps, event durations and future cash flows are absent. A 15-year horizon requires explicit scenarios and cannot be claimed as observed renewal performance.
No reliable Brier benefit over a fold-trained portfolio rate is established. Retention targeting and a production champion claim are unsupported.
Live Snowflake procedure compilation/execution remains unverified.
