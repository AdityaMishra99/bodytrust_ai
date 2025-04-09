#!/usr/bin/env python
# coding: utf-8

# # "Earn trust in health like you do in finance – with data."
# 
# ## Summary:
# BodyTrust AI is a data-driven platform that calculates a personalized "Fitness Trust Score" based on physical activity, habits, and wellness indicators — similar to how a credit score works in finance. By analyzing metrics like exercise frequency, intensity, sleep quality, hydration, nutrition, and even stress levels, BodyTrust AI helps users, trainers, and healthcare providers quantify and track overall health in a single score. With visual insights and actionable recommendations, this tool empowers people to build trust in their own health journey using real-time analytics and behavioral data.

# In[9]:


import pandas as pd
import numpy as np

# Load main datasets Amazon Mechanical Turk
activity_df = pd.read_csv("C:/Users/DELL/Downloads/archive (16)/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv")
calories_df = pd.read_csv("C:/Users/DELL/Downloads/archive (16)/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyCalories_merged.csv")
sleep_df = pd.read_csv("C:/Users/DELL/Downloads/archive (16)/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/sleepDay_merged.csv")
weight_df = pd.read_csv("C:/Users/DELL/Downloads/archive (16)/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/weightLogInfo_merged.csv")


# # Convert all date columns to proper format

# In[10]:


activity_df['ActivityDate'] = pd.to_datetime(activity_df['ActivityDate']).dt.date
calories_df['ActivityDay'] = pd.to_datetime(calories_df['ActivityDay']).dt.date
sleep_df['SleepDay'] = pd.to_datetime(sleep_df['SleepDay']).dt.date
weight_df['Date'] = pd.to_datetime(weight_df['Date']).dt.date


# # Merge All DataFrames
# We’ll merge them on Id and matching date columns.

# In[11]:


# Merge activity with calories
merged_df = pd.merge(activity_df, calories_df, left_on=['Id', 'ActivityDate'], right_on=['Id', 'ActivityDay'], how='left')

# Merge with sleep
merged_df = pd.merge(merged_df, sleep_df, left_on=['Id', 'ActivityDate'], right_on=['Id', 'SleepDay'], how='left')

# Merge with weight
merged_df = pd.merge(merged_df, weight_df, left_on=['Id', 'ActivityDate'], right_on=['Id', 'Date'], how='left')


# # Confirm Final Columns
# Run this to check what you’re working with:

# In[12]:


print(merged_df.columns.tolist())


# # Drop unnecessary or duplicate columns
# We'll keep only the useful features for trust scoring.

# In[13]:


merged_df_cleaned = merged_df.drop(columns=[
    'Calories_x', 'ActivityDay', 'SleepDay', 'Date', 'WeightPounds', 
    'IsManualReport', 'LogId', 'LoggedActivitiesDistance', 'TotalSleepRecords'
])


# # Handle Missing Values

# In[14]:


# Check for missing values
print(merged_df_cleaned.isnull().sum())

# Drop rows with too many nulls or fill as needed
merged_df_cleaned = merged_df_cleaned.dropna(subset=[
    'Calories_y', 'TotalMinutesAsleep', 'BMI'
])


# # Fill missing values smartly
# This gives us a bigger dataset but makes some assumptions:
# We can then ignore WeightKg and Fat for now, since they’re not needed for the trust score.

# In[15]:


# Fill missing sleep with median sleep
merged_df_cleaned['TotalMinutesAsleep'].fillna(merged_df_cleaned['TotalMinutesAsleep'].median(), inplace=True)

# Fill missing BMI with median BMI
merged_df_cleaned['BMI'].fillna(merged_df_cleaned['BMI'].median(), inplace=True)


# # Normalize Key Features
# We’ll bring everything onto the same scale (0 to 1) using MinMaxScaler, which is perfect for scoring models.

# In[16]:


from sklearn.preprocessing import MinMaxScaler

# Features to normalize
features_to_scale = [
    'TotalSteps', 'VeryActiveMinutes', 'SedentaryMinutes',
    'TotalMinutesAsleep', 'Calories_y', 'BMI'
]

scaler = MinMaxScaler()
merged_df_cleaned[features_to_scale] = scaler.fit_transform(merged_df_cleaned[features_to_scale])


# # Build the BodyTrust Score
# Let’s define the scoring logic: a weighted average of normalized values. We can tweak the weights based on what you want to emphasize — right now it’s equal weightage (20% each).

# In[17]:


# Scoring Logic
merged_df_cleaned['BodyTrustScore'] = (
    merged_df_cleaned['TotalSteps'] * 0.2 +
    merged_df_cleaned['VeryActiveMinutes'] * 0.2 +
    merged_df_cleaned['TotalMinutesAsleep'] * 0.2 +
    merged_df_cleaned['Calories_y'] * 0.2 +
    (1 - merged_df_cleaned['BMI']) * 0.2  # Lower BMI = better
) * 100  # Scale to 0–100


# # Preview the Trust Scores

# In[18]:


# Show top 10 people/days with the highest score
merged_df_cleaned[['Id', 'ActivityDate', 'BodyTrustScore']].sort_values(by='BodyTrustScore', ascending=False).head(10)


# # Visualize Insights (Data Storytelling)
# Let’s build some dope plots:
# ## Distribution of BodyTrust Scores

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.histplot(merged_df_cleaned['BodyTrustScore'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of BodyTrust AI Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ## Top 5 Users with Highest Avg Scores

# In[20]:


top_users = merged_df_cleaned.groupby('Id')['BodyTrustScore'].mean().sort_values(ascending=False).head(5)

top_users.plot(kind='bar', color='mediumseagreen')
plt.title('Top 5 Users by Average BodyTrust Score')
plt.xlabel('User ID')
plt.ylabel('Avg Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ## Time Series: How One User’s Score Changes Over Time
# Pick one user (e.g. 6962181067):

# In[21]:


user_data = merged_df_cleaned[merged_df_cleaned['Id'] == 6962181067]
user_data = user_data.sort_values('ActivityDate')

plt.plot(user_data['ActivityDate'], user_data['BodyTrustScore'], marker='o')
plt.title('Daily BodyTrust Score Over Time (User 6962181067)')
plt.xlabel('Date')
plt.ylabel('BodyTrust Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# # KMeans Clustering on Health Levels
# 
# Select features for clustering
# 
# Let’s pick:
# 
# 1. BodyTrustScore
# 
# 2. TotalSteps
# 
# 3. VeryActiveMinutes
# 
# 4. Calories_y
# 
# 5. TotalMinutesAsleep
# 
# 6. BMI

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Filter rows with no missing values for selected features
features = ['BodyTrustScore', 'TotalSteps', 'VeryActiveMinutes', 'Calories_y', 'TotalMinutesAsleep', 'BMI']
cluster_df = merged_df_cleaned[features].dropna()

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)


# ## Find the optimal number of clusters (Elbow Method)
# 
# Look for the “elbow point” in the plot (usually around k = 3 or 4). That’s our sweet spot.

# In[23]:


inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# ## Apply KMeans with chosen k (e.g. 3)

# In[24]:


kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to original DataFrame
cluster_df['Cluster'] = cluster_labels


# We just need to merge back the original columns (like Id and ActivityDate) from your main cleaned dataset (merged_df_cleaned) into the cluster_df after clustering.

# ## Assuming you built cluster_df like this:

# In[25]:


cluster_features = merged_df_cleaned[[
    'BodyTrustScore', 'Calories_y', 'TotalMinutesAsleep', 'BMI'
]].copy()


# ## And applied KMeans like:

# In[26]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df = cluster_features.copy()
cluster_df['Cluster'] = kmeans.fit_predict(cluster_features)


# ##  Now MERGE the original columns back:

# In[27]:


# Reset index to keep original reference
merged_ref = merged_df_cleaned[['Id', 'ActivityDate']].reset_index(drop=True)
cluster_df = cluster_df.reset_index(drop=True)

# Merge with original identifiers
cluster_df = pd.concat([merged_ref, cluster_df], axis=1)


# ## Visualize Clusters

# ### Add labels like:
# 
# 1. Cluster 0 = “Elite”
# 
# 2. Cluster 1 = “Moderate”
# 
# 3. Cluster 2 = “Needs Improvement”
# 
# ## Step-by-Step Plan
# Map Recommendations Based on Clusters We already have 3 clusters:
# Elite Performer
# 
# Average Active
# 
# Needs Improvement
# 
# Let’s define actionable, motivational, and specific recommendations for each one.
# 
# Add Recommendations to the DataFrame We’ll create a new column like Recommendation in cluster_df.
# 
# Preview and Display Top Recommendations

# In[28]:


cluster_df['HealthTier'] = cluster_df['Cluster'].map({
    0: 'Elite Performer',
    1: 'Average Active',
    2: 'Needs Improvement'
})
# Define cluster-based recommendations
recommendation_map = {
    'Elite Performer': "Maintain your routine! Consider advanced workouts and balanced nutrition.",
    'Average Active': "Good progress! Try increasing active minutes and improve sleep consistency.",
    'Needs Improvement': "Start with 20-30 mins of light activity daily. Improve sleep hygiene and track meals."
}

# Add recommendation column based on Health Tier
cluster_df['Recommendation'] = cluster_df['HealthTier'].map(recommendation_map)

# Preview the output
cluster_df[['Id', 'ActivityDate', 'BodyTrustScore', 'HealthTier', 'Recommendation']].head()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=cluster_df,
    x='BodyTrustScore',
    y='Calories_y',
    hue='HealthTier',   # <- Use HealthTier instead of Cluster
    palette='Set2',
    s=100
)
plt.title('BodyTrust Score vs Calories by Health Cluster')
plt.xlabel('BodyTrust Score')
plt.ylabel('Calories')
plt.grid(True)
plt.legend(title='Health Tier')  # Optional: Add title to legend
plt.show()


# ## visualize cluster centers or compare average values per cluster:

# In[30]:


# Define the features used for clustering
features = ['BodyTrustScore', 'Calories_y', 'TotalMinutesAsleep', 'BMI']

# Compute mean values by cluster tier
cluster_means = cluster_df.groupby('HealthTier')[features].mean()

# Display
print(cluster_means)


# ## Plotly Dash Code Using In-Memory DataFrame

# In[31]:


# Re-introduce TotalSteps for Plotly sizing
cluster_df = pd.merge(
    cluster_df,
    activity_df[['Id', 'ActivityDate', 'TotalSteps']],
    on=['Id', 'ActivityDate'],
    how='left'
)


# In[32]:


# Merge only the missing activity features into cluster_df
cluster_df = pd.merge(
    cluster_df,
    activity_df[['Id', 'ActivityDate', 'VeryActiveMinutes',
                 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']],
    on=['Id', 'ActivityDate'],
    how='left'
)


# In[33]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Select features and target
features = [
    'Calories_y', 'TotalMinutesAsleep', 'BMI',
    'TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
    'LightlyActiveMinutes', 'SedentaryMinutes'
]
target = 'BodyTrustScore'

X = cluster_df[features]
y = cluster_df[target]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


# ## Use our trained RandomForestRegressor to predict the future BodyTrust Score for each user based on their latest available data.
# 
# Group by user → Get their most recent record.
# 
# Select features from those records.
# 
# Run predictions using the trained model.
# 
# Add PredictedScore column back into the dataset.
# 
# (Optional) Calculate delta between predicted and current score.

# In[34]:


# 1. Get latest record for each user
latest_df = cluster_df.sort_values('ActivityDate').groupby('Id').tail(1)

# 2. Extract features from latest records
X_latest = latest_df[features]

# 3. Predict future BodyTrust Score
latest_df['PredictedScore'] = model.predict(X_latest)

# 4. Optional: Calculate change from current score
latest_df['ScoreDelta'] = latest_df['PredictedScore'] - latest_df['BodyTrustScore']

# 5. Preview predictions
latest_df[['Id', 'ActivityDate', 'BodyTrustScore', 'PredictedScore', 'ScoreDelta', 'HealthTier']]


# ## Merge PredictedScore and ScoreDelta into main dataframe

# In[35]:


cluster_df = pd.merge(
    cluster_df,
    latest_df[['Id', 'PredictedScore', 'ScoreDelta']],
    on='Id',
    how='left'
)


# In[36]:


cluster_df[['Id', 'BodyTrustScore', 'PredictedScore', 'ScoreDelta']].head()


# ## Export to CSV

# In[37]:


cluster_df.to_csv("bodytrust_ai_final.csv", index=False)
print("✅ File saved as 'bodytrust_ai_final.csv'")

