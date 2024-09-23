#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('CDR-Call-Details.csv')


# In[3]:


df


# Phase 1:

# In[4]:


df.head


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[10]:


df.value_counts


# In[12]:


df.shape


# Phase 2

# In[14]:


missing_values = df.isnull().sum()


# In[15]:


missing_values


# In[25]:


duplicated_rows = df.duplicated().sum()


# In[26]:


duplicated_values


# In[27]:


summary_statistics = df.describe()


# In[28]:


summary_statistics


# In[36]:


print('Missing values:',missing_values)
print('Duplicated rows:',duplicated_rows)
print('Summary Statistics:\n',summary_statistics)


# In[37]:


df.fillna(df.mean(), inplace = True)


# In[38]:


df.drop_duplicates(inplace=True)


# In[39]:


print(df.info())


# Phase 2: Data Exploration and Visualization

# In[40]:


print(df.describe())


# In[42]:


customer_groups = df.groupby('Phone Number').mean()


# In[43]:


customer_groups


# In[46]:


sns.countplot(x = 'Churn', data=df)
plt.title('Churn Distribution')
plt.show()


# In[47]:


sns.boxplot(x = 'Churn', y = 'Account Length', data = df)
plt.title('Churn vs Account length of customer call')
plt.show()


# In[53]:


plt.figure(figsize = (10,15))
sns.boxplot(data = df)
plt.title('Outlier detection')
plt.show()


# Phase 3: Data Analysis and Feature Engineering

# In[56]:


columns_to_analyze = [
    'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
    'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 
    'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls', 'Churn'
]


for column in columns_to_analyze:
    plt.figure(figsize = (7,4))
    sns.histplot(df[column], kde = True)
    plt.title('Distribution of the column')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[59]:


# Bivariate analysis between Churn and other features
for column in columns_to_analyze[:-1]:  # Exclude 'Churn' itself
    plt.figure(figsize=(8, 5))
    
    if df[column].dtype == 'object':
        # If the column is categorical, use a count plot
        sns.countplot(x=column, hue='Churn', data=df)
    else:
        # For numerical columns, use boxplot
        sns.boxplot(x='Churn', y=column, data=df)
    
    plt.title(f"{column} vs Churn")
    plt.show()


# In[60]:


df['Callratio'] = df['Day Calls'] / df['Day Mins']
df['AverageCallDuration'] = df['Day Mins'] / df['Day Calls']


# In[64]:


print(df[['Callratio','AverageCallDuration']].head())


# Phase 4: Predictive Modeling and Recommendations

# In[65]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[66]:


X = df.drop(columns = ['Churn'])
y = df['Churn']


# In[67]:


X


# In[68]:


y


# In[76]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


X = df.drop(columns=['Churn', 'Phone Number'])


categorical_columns = X.select_dtypes(include=['object']).columns


le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string before encoding to handle NaNs


imputer = SimpleImputer(strategy='mean')


X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Model selection - Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[77]:


y_pred = model.predict(X_test)


# In[78]:


y_pred


# In[82]:


accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# In[81]:


print(f"Accuracy: {accuracy}")


# In[83]:


from sklearn.model_selection import GridSearchCV


# In[88]:


# Hyperparameter tuning (example using GridSearchCV)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train, y_train)



# Phase 5: Reporting and Documentation

# In[92]:


print(f"Model Accuracy : {accuracy}")


# In[97]:


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title('Feature Imporatances of the Churn Analysis')
plt.bar(range(X.shape[1]), importances[indices], align = 'center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation = 90)
plt.show()


# In[95]:


with open('churn_analysis_report.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy}\n")
    f.write("Feature Importances:\n")
    for i in indices:
        f.write(f"{X.columns[i]}: {importances[i]}\n")


# Phase 6: Real-World Applications

# In the context of telecom churn prediction, the insights generated from the predictive model, along with data analysis, can be directly applied to tackle several key business challenges. The primary objective here is to reduce customer churn, which significantly impacts telecom revenue, while also enhancing customer service quality. Here's how these insights can be applied:
# 
# Targeted Customer Retention Programs:
# 
# Insights from the churn model: The model has identified key factors contributing to churn, such as high total minutes usage, frequent customer service calls, or high international call charges. These variables can be used to flag customers at high risk of leaving.
# Application: The telecom company can create customer retention programs specifically targeting high-risk customers. For example, customers who have called customer service multiple times may be offered personalized service plans, discounts, or loyalty rewards to enhance their satisfaction and prevent them from switching providers.
# Action Plan: Use the predicted churn probability to segment customers into risk groups. Proactively reach out to high-risk customers before they churn, offering incentives or improved services.
# Improved Service Quality:
# 
# Insights from the churn model: High customer service call counts correlate with churn, indicating that unresolved issues might be a significant factor.
# Application: To improve customer service, telecom operators could analyze the reasons behind frequent service calls, addressing common complaints and implementing better self-service tools or AI-powered chatbots to resolve issues faster. Identifying specific services or locations that generate the most complaints could lead to infrastructure or policy improvements.
# Action Plan: Focus on improving response times and customer satisfaction metrics in areas with higher churn rates. This could involve investing in customer care training, expanding service offerings, or introducing more reliable service in regions with network problems.
# Dynamic Pricing Models:
# 
# Insights from the churn model: High usage charges in specific categories (e.g., international calls or data) can contribute to customer churn.
# Application: Implement dynamic pricing models where customers at risk of churn are offered personalized discounts or pricing adjustments for international or overage charges. This approach can be particularly effective if you can segment customers who regularly exceed their plan limits but don’t require an unlimited plan.
# Action Plan: Design flexible pricing plans based on customer usage patterns to encourage long-term engagement and satisfaction, offering benefits such as discounted international minutes for those making frequent international calls.
# Service Plan Optimization:
# 
# Insights from the churn model: Customers who use a lot of minutes but are still dissatisfied could be on the wrong plan.
# Application: Offer tailored service plans to customers based on their actual usage patterns (as predicted by the model). If the model identifies customers using excessive daytime minutes, for instance, they could be nudged to switch to a more cost-effective plan with higher daytime limits. This approach reduces bill shock and enhances the perception of value.
# Action Plan: Develop automated systems that recommend optimal plans for users based on their behavior, sending periodic recommendations for plan upgrades or downgrades to match customer needs.

# Project Conclusion and Future Steps

# Churn Reduction KPIs:
# 
# Measuring the impact: The primary metric for assessing the success of churn prevention initiatives is the churn rate itself. By implementing the churn prediction model, the company can track changes in the churn rate before and after intervention. This can be expressed as a percentage reduction in churn.
# Expected impact: A well-targeted churn prevention strategy could reduce churn rates by 5-10%, leading to significant improvements in customer retention and long-term revenue.
# Customer Lifetime Value (CLV):
# 
# Impact on revenue: Retaining customers increases their Customer Lifetime Value (CLV), which is the net profit attributed to the entire future relationship with a customer. By preventing churn, the telecom company ensures that they benefit from the long-term revenue generated by existing customers rather than having to spend excessively on acquiring new ones.
# Expected impact: A decrease in churn could lead to an increase in average CLV, which can be modeled and projected over time to forecast the financial benefit of the churn reduction initiatives.
# Cost Savings from Customer Service Optimization:
# 
# Insights application: By identifying patterns in frequent customer service complaints, the company can reduce the operational costs of customer service by addressing common issues proactively (e.g., resolving network issues in a specific region).
# Expected impact: A reduction in customer service calls can directly lead to cost savings for the company and improved customer satisfaction. For example, if the model identifies that frequent international callers often churn due to high charges, creating better international plans can reduce both complaints and churn.
# Improved Customer Satisfaction Scores:
# 
# Customer satisfaction: Reducing churn through tailored retention strategies and improved service will likely result in higher customer satisfaction (CSAT) and Net Promoter Scores (NPS), which are key indicators of customer loyalty.
# Expected impact: The churn prediction model can help focus efforts on the most dissatisfied customers, leading to a rise in overall satisfaction metrics. Improved customer experiences, in turn, lead to better word-of-mouth marketing and organic growth.
# Strategic Decision-Making and Network Investments:
# 
# Better resource allocation: Insights from the churn model help the company make strategic decisions about where to allocate resources. For example, if churn is high in specific geographic areas, this may indicate network issues or lack of customer support in those regions.
# Expected impact: By using predictive models, telecom operators can prioritize investments in network infrastructure, marketing, and customer service improvements in areas that are most likely to experience churn. This leads to more efficient use of resources and greater returns on investment.
# Summary
# The Telecom Churn Prediction model provides actionable insights that telecom companies can use to improve customer retention and optimize service quality. By identifying high-risk customers, targeting retention efforts, offering personalized plans, and improving customer service, telecom operators can significantly reduce churn rates. In addition, the data-driven recommendations enhance overall operational efficiency, leading to better customer satisfaction and increased revenue. The impact of these initiatives can be measured through key metrics such as churn reduction, CLV growth, cost savings, and improved customer satisfaction.

# In[ ]:




