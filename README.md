# careloop-no-show-guard
# CareLoop Clinics — No-Show Guard
**Repository:** <PASTE_YOUR_PUBLIC_GITHUB_REPO_LINK_HERE>

## Executive Summary
* **Part 1 (Data Cleaning):** Cleaned 1,020 raw bookings down to 990 valid appointments (removing 20 duplicates, 6 invalid dates, and 4 blank statuses), revealing an overall no-show rate of **22.0%** (218 no-shows).
* **Part 2 (SQL Analysis):** No-shows spike sharply with booking lead time—rising from **16.2%** (1–7 days) to **30.0%** (15–30 days) and **60.5%** (31+ days)—and cluster highest at Hadapsar Care Centre (**27.3%**, 143 bookings) among high-volume clinics and Gomti Nagar Care Point (**33.3%**, 90 bookings) overall.
* **Part 3 (Risk Modeling):** Logistic Regression achieved the highest 5-fold cross-validated ROC-AUC (**0.77**) and test ROC-AUC (**0.789**); in this model, holding other features constant, sending a reminder is associated with **69% lower odds** of a no-show (`exp(coef/scale) = 0.31`).
* **Part 4 (GenAI Follow-Up Review):** Structured extraction on 10 call-centre notes (9 non-blank, 1 skipped blank) showed that **2 of 2** `forgot_or_no_reminder` cases occurred on appointments where `reminder_sent = No`.

## Recommendation
CareLoop should first guarantee automated reminders for all bookings made **15+ days** in advance (where no-show rates jump from **16.2%** to **30.0%–60.5%**) and deploy the evening-before Logistic Regression risk score at the **0.35** threshold to guide front-desk calls. In our model, reminders are associated with **69% lower no-show odds**, and **100% (2/2)** of the `forgot_or_no_reminder` notes had no reminder sent. Prioritizing calls to flagged high-risk patients—especially at high-no-show clinics like Hadapsar (**27.3%**) and Malviya Nagar (**26.8%**) and for patients with prior no-shows—will capture **57.1%** of upcoming no-shows while keeping evening call volume manageable (~25% of bookings).

## Model Selection & Threshold
* **Chosen Model:** Logistic Regression (`Pipeline` with `StandardScaler` + `LogisticRegression(max_iter=1000)`), selected strictly on 5-fold training cross-validated ROC-AUC (**0.77**), outperforming Decision Tree depth-3 (**0.68**), default Random Forest (**0.75**), and tuned Random Forest. On the single test evaluation, it achieved **ROC-AUC = 0.789**.
* **Chosen Threshold:** **0.35** (selected via out-of-fold training F1: **0.489** vs. **0.347** at 0.50). On the test set at 0.35, it achieved **57.1% recall** and **57.1% precision** (49 bookings flagged). In production, the threshold should be tuned to front-desk evening call capacity against the cost of an empty slot.

## Limitations
1. **Small GenAI sample:** The 10 analysed call-centre notes are an illustration and cannot represent clinic-wide root-cause distributions.
2. **Pre-split cleaning & random split:** Medians and IQR fences were computed on all 990 rows before the train/test split, and the split is random rather than past-to-future (allowing the same patient to appear in both train and test).
3. **Observational data:** The dataset is observational, so model coefficients (such as the reminder effect) show associations rather than proven causation.
