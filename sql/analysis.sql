-- CreditIQ: MySQL Exploratory Data Analysis & Credit Risk Queries

-- 1. Default Rate by Loan Grade
SELECT 
    grade,
    sub_grade,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END) AS defaults,
    ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0.0 END) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_interest_rate,
    ROUND(AVG(loan_amnt), 0) AS avg_loan_amount
FROM lending_club_loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
GROUP BY grade, sub_grade
ORDER BY grade, sub_grade;

-- 2. Annual Vintage Default Trajectory
SELECT 
    YEAR(STR_TO_DATE(issue_d, '%b-%Y')) AS issue_year,
    COUNT(*) AS total_loans,
    ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0.0 END) * 100, 2) AS default_rate_pct,
    ROUND(AVG(dti), 2) AS avg_dti,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico
FROM lending_club_loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
GROUP BY issue_year
ORDER BY issue_year;

-- 3. High Risk Borrower Profiling (Affordability vs Delinquency)
SELECT 
    CASE 
        WHEN (loan_amnt / annual_inc) > 0.35 THEN 'High DTI (>35%)'
        WHEN (loan_amnt / annual_inc) BETWEEN 0.20 AND 0.35 THEN 'Moderate DTI (20-35%)'
        ELSE 'Healthy DTI (<20%)'
    END AS affordability_segment,
    COUNT(*) AS borrower_count,
    ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0.0 END) * 100, 2) AS default_rate_pct,
    ROUND(AVG(revol_util), 2) AS avg_revol_util
FROM lending_club_loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
  AND annual_inc > 0
GROUP BY affordability_segment
ORDER BY default_rate_pct DESC;

