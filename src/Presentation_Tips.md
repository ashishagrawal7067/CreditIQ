## Questions They Will Ask You
1. **Why did you choose Gini coefficient over accuracy?**

Accuracy is misleading on imbalanced datasets. If 80% of loans are fully paid, a model that predicts "fully paid" for every loan achieves 80% accuracy while being completely useless. Gini coefficient — which equals 2×AUC−1 — measures how well the model separates defaulters from non-defaulters regardless of class balance. It's also the metric credit risk teams actually use in production, so reporting it signals you understand the industry.

2. **Why did you drop Current and Late loans from your training data?**

Their outcome is unknown — we don't yet know if they'll be fully paid or charged off. Training on them would introduce label noise. More importantly, including them would be a subtle form of data leakage because in production you'd never have that information at origination. Dropping them keeps the training set clean and the problem well-defined.

3. **Your model only approves 31.5% of loans at the best threshold — isn't that too restrictive for a real lender?**

This is a business decision, not a modelling decision. The threshold is a dial. A lender with aggressive growth targets sets it higher — accepting more default risk in exchange for volume. A lender focused on portfolio quality sets it lower. My simulation shows the full trade-off curve across six thresholds so a business can choose based on their strategy. The model doesn't make that call — it just makes the consequences of each choice transparent.

4. **Why XGBoost over Random Forest given the Gini difference is small — only 0.0196?**

Two reasons. First, XGBoost trained in 93 seconds versus Random Forest's 452 seconds — nearly 5x faster, which matters enormously at production scale with daily retraining. Second, XGBoost's gradient boosting framework handles class imbalance more elegantly via scale_pos_weight, and it has more regularisation levers to prevent overfitting. The Gini gap is small but XGBoost wins on every other dimension.

5. **What would you do differently if you had more time?**

Three things. First, survival analysis — instead of binary default prediction, model time to default using Cox regression, which gives richer information for loan pricing. Second, reject inference — our model only trains on approved loans, but a real lender needs to estimate default risk for declined applicants too, which requires specialised techniques like augmentation or parcelling.

6. **How would you put this model into production?**

The XGBoost model is serialised with joblib and can be wrapped in a FastAPI endpoint that accepts loan application features and returns a default probability in milliseconds. The credit scorecard from notebook 04 provides a parallel interpretable output for regulatory compliance. I'd retrain monthly on new originations, monitor Gini on a rolling holdout, and alert if it drops more than 3 points — that's a standard model governance threshold. I know right?

>The complete project: We have four notebooks, 15 charts, two saved models, a credit scorecard, a business simulation with a boardroom-ready headline number, and a professional README.
