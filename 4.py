from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Locate the CSV in the same folder as this script.
folder = Path(__file__).resolve().parent
df = pd.read_csv("/Users/timersenarith/Desktop/Northeastern University/ALY 6105 Applied Statistics/R/Telco_customer_churn.csv")

# Inspect the dataset.
print(df.head())
print(df.columns.tolist())
print(df.shape)
print(df.dtypes)

# EDA 
# Total Number of Customers
total_customers = len(df) 
print(total_customers)

# Calculate mean, median and standard deviation of MonthlyChargers
print(df['Monthly Charges'].agg(['mean', 'median', 'std']))

# Calculate mean, median and standard deviation of tenure
print(df['Tenure Months'].agg(['mean', 'median', 'std']))

# Histogram and Boxplot for monthly charges
plt.hist(df['Monthly Charges'], edgecolor='black')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.title('Distribution of Monthly Charges')
plt.show()

plt.boxplot(df['Monthly Charges'])
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.title('Distribution of Monthly Charges')
plt.show()

# Table Formatting
def show_table(data, title):
    plt.figure(figsize=(9, 4))
    plt.axis('off')

    table = plt.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title(title)
    plt.show()

# 1. Overall Churn
total_customers = len(df)
churned = (df['Churn Label'] == 'Yes').sum()
did_not_churn = (df['Churn Label'] == 'No').sum()

table1 = pd.DataFrame({
    'Category': ['Total Customers', 'Churned', 'Did Not Churn'],
    'Count': [total_customers, churned, did_not_churn]
})

show_table(table1, 'Table 1: Overall Churn')

# 2. Gender x Churn
table2 = pd.crosstab(
    df['Gender'],
    df['Churn Label'],
    margins=True,
    margins_name='Total'
)
show_table(table2.reset_index(), 'Table 2: Gender x Churn')

# 3. Partner Status x Churn 
partner_status = df['Partner'].map({
    'Yes': 'Has Partner',
    'No': 'No Partner'
})

table3 = pd.crosstab(
    partner_status,
    df['Churn Label'],
    margins=True,
    margins_name='Total'
)
show_table(table3.reset_index(), 'Table 3: Partner Status x Churn')

# 4. Dependents x Churn
dependents_status = df['Dependents'].map({
    'Yes': 'Has Dependents',
    'No': 'No Dependents'    
})

table4 = pd.crosstab(
    dependents_status,
    df['Churn Label'],
    margins=True,
    margins_name='Total'
)
    
show_table(table4.reset_index(), 'Table 4: Dependents x Churn')

# 5. Monthly Charges Category x Churn
# Check for missing charges before assigning categories.
if df['Monthly Charges'].isna().any():
    raise ValueError('Check missing monthly charges before continuing.')

# Monthly Categories (High > 80$)
df['Charge Category'] = 'Not High'
df.loc[df['Monthly Charges'] > 80, 'Charge Category'] = 'High'

table5 = pd.crosstab(
    df['Charge Category'],
    df['Churn Label'],
    margins=True,
    margins_name='Total'
)
show_table(
    table5.reset_index(), 'Table 5: Monthly Charges Category x Churn')

# 6. Gender x Partner x Churn 
table6 = pd.crosstab(
    [df['Gender'], partner_status],
    df['Churn Label']
)

table6['Total'] = table6.sum(axis=1)

show_table(
    table6.reset_index(),
    'Table 6: Gender × Partner × Churn'
)

# 4.1 Marginal Probability 
p_churn = churned / total_customers
print('P(Churn):', round(p_churn, 4))

# 4.2 Conditional Probabilities
# 4.2a Gender and Churn
male_churned = table2.loc['Male', 'Yes']
male_total = table2.loc['Male', 'Total']

female_churned = table2.loc['Female', 'Yes']
female_total = table2.loc['Female', 'Total']

p_churn_male = male_churned / male_total
p_churn_female = female_churned / female_total

print('\n4.2a: Gender and Churn')
print(f'Male: {male_churned}/{male_total} = {p_churn_male:.4f}')
print(f'Female: {female_churned}/{female_total} = {p_churn_female:.4f}')

gender_difference = abs(p_churn_female - p_churn_male) * 100
print(f'Difference: {gender_difference:.2f} percentage points')

# 4.2b Partner Status and Churn
partner_churned = table3.loc['Has Partner', 'Yes']
partner_total = table3.loc['Has Partner', 'Total']

no_partner_churned = table3.loc['No Partner', 'Yes']
no_partner_total = table3.loc['No Partner', 'Total']

p_churn_partner = partner_churned / partner_total
p_churn_no_partner = no_partner_churned / no_partner_total

print('\n4.2b: Partner Status and Churn')
print(f'Has Partner: {partner_churned}/{partner_total} = {p_churn_partner:.4f}')
print(f'No Partner: {no_partner_churned}/{no_partner_total} = {p_churn_no_partner:.4f}')

partner_difference = abs(p_churn_partner - p_churn_no_partner) * 100
print(f'Difference: {partner_difference:.2f} percentage points')

if p_churn_partner > p_churn_no_partner:
    print('Customers with a partner have a higher churn probability.')
elif p_churn_partner < p_churn_no_partner:
    print('Customers without a partner have a higher churn probability.')
else:
    print('Both groups have the same churn probability.')

# 4.2c Dependents and Churn
dependents_churned = table4.loc['Has Dependents', 'Yes']
dependents_total = table4.loc['Has Dependents', 'Total']

no_dependents_churned = table4.loc['No Dependents', 'Yes']
no_dependents_total = table4.loc['No Dependents', 'Total']

p_churn_dependents = dependents_churned / dependents_total
p_churn_no_dependents = no_dependents_churned / no_dependents_total

print('\n4.2c: Dependents and Churn')
print(f'Has Dependents: {dependents_churned}/{dependents_total} = {p_churn_dependents:.4f}')
print(f'No Dependents: {no_dependents_churned}/{no_dependents_total} = {p_churn_no_dependents:.4f}')

dependents_difference = abs(p_churn_dependents - p_churn_no_dependents) * 100
print(f'Difference: {dependents_difference:.2f} percentage points')

if p_churn_dependents > p_churn_no_dependents:
    print('Customers with dependents have a higher churn probability.')
elif p_churn_dependents < p_churn_no_dependents:
    print('Customers without dependents have a higher churn probability.')
else:
    print('Both groups have the same churn probability.')

# 4.3 Joint Probability and the Multiplication Rule
female_partner_churned = table6.loc[('Female', 'Has Partner'), 'Yes']
female_partner_total = table6.loc[('Female', 'Has Partner'), 'Total']

p_joint = female_partner_churned / total_customers

print('\n4.3: Joint Probability')
print(f'{female_partner_churned}/{total_customers} = {p_joint:.4f}')
print(f'This segment represents {p_joint * 100:.2f}% of all customers.')

# Multiplication Rule
p_female = female_total / total_customers
p_partner_given_female = female_partner_total / female_total
p_churn_given_female_partner = female_partner_churned / female_partner_total

p_multiplication = (
    p_female
    * p_partner_given_female
    * p_churn_given_female_partner
)

print(
    f'({female_total}/{total_customers}) × '
    f'({female_partner_total}/{female_total}) × '
    f'({female_partner_churned}/{female_partner_total}) '
    f'= {p_multiplication:.4f}'
)

# 4.4 Addition Rule
high_total = table5.loc['High', 'Total']
high_churned = table5.loc['High', 'Yes']

p_churn = churned / total_customers
p_high = high_total / total_customers
p_churn_and_high = high_churned / total_customers

p_churn_or_high = p_churn + p_high - p_churn_and_high

print('\n4.4: Addition Rule')
print(f'P(Churn): {churned}/{total_customers} = {p_churn:.4f}')
print(f'P(High): {high_total}/{total_customers} = {p_high:.4f}')
print(f'P(Churn AND High): {high_churned}/{total_customers} = {p_churn_and_high:.4f}')

print(
    f'P(Churn OR High): '
    f'{churned}/{total_customers} + '
    f'{high_total}/{total_customers} - '
    f'{high_churned}/{total_customers} '
    f'= {p_churn_or_high:.4f}'
)

print(f'Percentage: {p_churn_or_high * 100:.2f}%')

# 4.5 Bayes' Theorem
p_churn_given_high = high_churned / high_total

p_high_given_churn_bayes = (
    p_churn_given_high * p_high
) / p_churn

print('\n4.5: Bayes Theorem')
print(f'P(Churn | High): {high_churned}/{high_total} = {p_churn_given_high:.4f}')

print(
    f'Bayes calculation: '
    f'[({high_churned}/{high_total}) × '
    f'({high_total}/{total_customers})] / '
    f'({churned}/{total_customers}) '
    f'= {p_high_given_churn_bayes:.4f}'
)

# Verify directly from the frequency table
p_high_given_churn_direct = high_churned / churned

print(
    f'Direct calculation: {high_churned}/{churned} '
    f'= {p_high_given_churn_direct:.4f}'
)

# Allow for tiny differences from computer arithmetic
match = abs(p_high_given_churn_bayes - p_high_given_churn_direct) < 0.0000000001
print('Do the answers match?', match)