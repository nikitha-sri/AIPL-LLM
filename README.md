This project is a machine learning-based IPL match forecasting system developed for the AIPL 2026 Forecast Competition. The model predicts the probability of four possible match outcome classes using historical IPL ball-by-ball data from 2008–2025.

The solution focuses on minimizing Mean Columnwise Log Loss using calibrated probability predictions.

Competition Objective

For every IPL match, the model predicts probabilities for:

A_small → Team A wins by ≤ 20 runs OR ≤ 5 wickets
A_big → Team A wins by > 20 runs OR ≥ 6 wickets
B_small → Team B wins by ≤ 20 runs OR ≤ 5 wickets
B_big → Team B wins by > 20 runs OR ≥ 6 wickets

The final probabilities:

must sum to 1
contain no missing values
remain within the range [0,1]
Dataset

Historical IPL ball-by-ball data:

Seasons: 2008–2025
1100+ matches
272K+ deliveries

Files used:

train_IPL.csv
schedule.csv
sample_submission.csv
public_lb_matches.csv
Features Engineered
Team Strength Features
Team win percentage
Recent form (last 5 matches)
ELO ratings
Team strength difference
Venue Features
Average first innings score
Venue chase bias
Venue scoring factor
Toss Features
Toss winner
Toss decision
Toss advantage indicators
Match Context Features
Home/Away dynamics
Batting order
Historical performance trends
Machine Learning Model
Primary Model
CatBoostClassifier
Why CatBoost?

CatBoost performs exceptionally well on:

categorical cricket data
limited structured datasets
probability calibration tasks
multiclass classification problems
Model Optimization Techniques
Probability smoothing
Feature normalization
Overfitting control
Regularized tree depth
Validation-based tuning
Log-loss optimization
Validation Strategy

The model uses:

historical seasons for training
recent season data for validation

This helps simulate real-world IPL forecasting conditions.

Evaluation Metric

Competition Metric:

Mean Columnwise Log Loss

Lower score indicates better calibrated probabilities.

Technologies Used
Python
Pandas
NumPy
CatBoost
Scikit-learn
Output

The model generates:

submission.csv

Format:

match_id	A_big	A_small	B_big	B_small
How to Run
Install Dependencies
pip install pandas numpy scikit-learn catboost
Run Notebook

Execute all notebook cells in Kaggle or Jupyter Notebook.

The final submission file will automatically generate as:

submission.csv
Project Highlights
End-to-end IPL forecasting pipeline
Advanced feature engineering
ELO-based team ranking system
Venue-aware prediction system
Optimized for Kaggle leaderboard performance
Production-style ML workflow
Future Improvements

Potential upgrades:

Player-level embeddings
Deep learning ensemble models
Bayesian probability calibration
Real-time squad impact modeling
Weather and pitch condition integration
Author

Developed for the AIPL 2026 IPL Match Forecast Competition using machine learning and probabilistic sports analytics.
