# Customer Churn Probability Analysis

A Python project exploring customer churn through descriptive statistics, visualizations, and probability calculations.

Completed for ALY6105 Applied Statistics at Northeastern University.

## Overview

The analysis examines how churn varies across gender, partner status, dependent status, and monthly charges using a course-provided IBM Telco Customer Churn dataset containing 7,043 customers.

Monthly charges above $80 are classified as high.

## Methods

- Calculate descriptive statistics for monthly charges and tenure.
- Visualize monthly charges using a histogram and boxplot.
- Build frequency tables comparing customer groups with churn.
- Calculate marginal, conditional, and joint probabilities.
- Apply the multiplication rule, addition rule, and Bayes' theorem.
- Verify the Bayes result against a direct calculation.

## Key findings

Results from the assignment analysis:

| Customer group | Churn rate |
|---|---:|
| All customers | 26.54% |
| Male | 26.16% |
| Female | 26.92% |
| Has a partner | 19.66% |
| No partner | 32.96% |
| Has dependents | 6.52% |
| No dependents | 32.55% |
| Monthly charges above $80 | 33.98% |

Churn rates differed little by gender. Larger differences appeared across partner and dependent status.

Among customers who churned, 48.48% had high monthly charges. This differs from the 33.98% churn rate among high-charge customers because the denominators represent different groups.

These findings describe associations and do not establish causation.

## Tools

- Python
- Pandas
- Matplotlib

## Run the analysis

1. Obtain the course dataset, Telco_customer_churn.csv.
2. Update the pd.read_csv path in 4.py to match the dataset location on your computer.
3. Install dependencies: python3 -m pip install pandas matplotlib
4. Run the script: python3 4.py

The script prints statistics and probability calculations and displays charts and frequency tables. Close each plot window to continue if execution pauses.

The dataset and written report are not included in this repository.

## Limitations

The analysis compares customer groups without controlling for other characteristics. Further analysis could examine tenure, contract type, services, and monthly charges together to better understand the observed relationships.

## Author

Timer Senarith
