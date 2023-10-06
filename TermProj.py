import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader.iex import stats
from scipy.stats import f_oneway, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             silhouette_score, roc_curve)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import warnings
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:,.3f}".format

np.random.seed(5525)
`
# Load each dataset
url_athelete = 'https://raw.githubusercontent.com/mahimasingh-VTU/OlympicRankPredictor/main/Dataset/olympic_athletes.csv'
url_hosts = 'https://raw.githubusercontent.com/mahimasingh-VTU/OlympicRankPredictor/main/Dataset/olympic_hosts.csv'
url_medals = 'https://raw.githubusercontent.com/mahimasingh-VTU/OlympicRankPredictor/main/Dataset/olympic_medals.csv'
url_results = 'https://raw.githubusercontent.com/mahimasingh-VTU/OlympicRankPredictor/main/Dataset/olympic_results.csv'

athletes_df = pd.read_csv(url_athelete)
hosts_df = pd.read_csv(url_hosts)
medals_df = pd.read_csv(url_medals)
results_df = pd.read_csv(url_results)

# -------------------------------------------------------------------------------------------------------
# --------------------------------Cleaning Datasets-------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# check for missing values
# athelete dataset
missing_values_ath = athletes_df.isnull().sum().sum()
# Calculate the total number of entries in the dataset
total_ath = athletes_df.count().sum()
# Calculate the percentage of missing values
percent_missing_ath = (missing_values_ath / total_ath) * 100
# Print the percentage of missing values
print("Missing data percentage in athelete Dataset: ", percent_missing_ath, "%")

# Calculate the total number of missing values
missing_values_med = medals_df.isnull().sum().sum()
# Calculate the total number of entries in the dataset
total_med = medals_df.count().sum()
# Calculate the percentage of missing values
percent_missing_med = (missing_values_med / total_med) * 100
print("Missing data percentage in medals data set: ", percent_missing_med, "%")

# Calculate the total number of missing values
missing_values_host = hosts_df.isnull().sum().sum()
# Calculate the total number of entries in the dataset
total_host = hosts_df.count().sum()
# Calculate the percentage of missing values
percent_missing_host = (missing_values_host / total_host) * 100
print("Missing data percentage in hosts data set: ", percent_missing_host, "%")

# Calculate the total number of missing values
missing_values_res = results_df.isnull().sum().sum()
# Calculate the total number of entries in the dataset
total_res = results_df.count().sum()
# Calculate the percentage of missing values
percent_missing_res = (missing_values_res / total_res) * 100

# Results Dataset
print("Missing data percentage in results data set: ", percent_missing_res, "%")

# Fill missing values in 'first_game' with the mode
mode_first_game = athletes_df['first_game'].mode()[0]
athletes_df['first_game'].fillna(mode_first_game, inplace=True)

# Fill missing values in 'athlete_year_birth' with the mean
mean_year_birth = athletes_df['athlete_year_birth'].mean()
athletes_df['athlete_year_birth'].fillna(mean_year_birth, inplace=True)

# Drop the 'athlete_medals' and 'bio' columns entirely since 80 percent of them are null
athletes_df.drop(['athlete_medals', 'bio'], axis=1, inplace=True)

# Print the updated dataframe
print("Missing data percentage in athelete dataset after cleaning: ",
      (athletes_df.isnull().sum().sum() / athletes_df.count().sum()) * 100, "%")

# medals dataset

# filling values : cleaning and removing rows:

# Drop the 'athlete_medals' and 'bio' columns entirely
medals_df.drop(['participant_title', 'athlete_url', 'country_code'], axis=1, inplace=True)

# fill names if name not available
medals_df['athlete_full_name'] = medals_df['athlete_full_name'].fillna('Name Not Available')
# changing column name and extracting game year and location from slug game
medals_df['game_location'] = medals_df['slug_game'].apply(lambda x: x.split('-')[0])

medals_df['game_year'] = medals_df['slug_game'].apply(lambda x: x.split('-')[-1])

# medals_df = medals_df.drop('slug_game', axis=1)

# Print the updated dataframe
print("Missing data percentage in medals dataset after cleaning: ",
      (medals_df.isnull().sum().sum() / medals_df.count().sum()) * 100, "%")

# hosts datasets
results_df.drop(['athletes', 'rank_equal', 'value_unit', 'value_type', 'medal_type'], axis=1, inplace=True)
results_df.dropna(subset=['country_code', 'athlete_full_name', 'athlete_url'], inplace=True)
results_df['athlete_full_name'] = results_df['athlete_full_name'].fillna('Name Not Available', axis=0)

results_df['rank_position'] = results_df['rank_position'].fillna(results_df['rank_position'].mode()[0], axis=0)
results_df['game_year'] = results_df['slug_game'].apply(lambda x: x.split('-')[-1])

# Print the updated dataframe
print("Missing ata percentage after cleaning results dataset: ",
      (results_df.isnull().sum().sum() / results_df.count().sum()) * 100, "%")

# picking only year of first game and dropping rest string(country)
athletes_df['first_game_year'] = athletes_df['first_game'].apply(lambda x: x.split(" ")[-1])
# changing float to int
athletes_df['athlete_year_birth'] = athletes_df['athlete_year_birth'].astype(int)

# changing first game year fromobject to int
athletes_df['first_game_year'] = athletes_df['first_game_year'].astype(int)

# calculating their age
athletes_df['athletes_age'] = athletes_df['first_game_year'] - athletes_df['athlete_year_birth']

print("Unique values of atheletes age: ", athletes_df['athletes_age'].unique())

# Atheletes ages is given as negative hence deleting the data

athletes_df = athletes_df.drop(index=athletes_df[athletes_df['athletes_age'] < 0].index, axis=0)

# verifying the value
# print(athletes_df['athletes_age'].sort_values(ascending=True))
sns.displot(x=athletes_df['athletes_age'], bins=100, kde=True)
plt.ylim(0, 25)
# outliers r there hence revisiting age ,lets consider athletes age between 12-85 age
athletes_df = athletes_df.drop(
    index=athletes_df[(athletes_df['athletes_age'] > 85) | (athletes_df['athletes_age'] < 12)].index, axis=0)

# verify
plt.figure()
sns.displot(x=athletes_df['athletes_age'], bins=100, kde=True)
plt.ylim(0, 25)
plt.show()

# cleaning hosts dataset------------
# Let's Convert 'game_end_date' and 'game_start_date' into datetime object

hosts_df['game_end_date'] = hosts_df['game_end_date'].apply(lambda x: pd.to_datetime(x))
hosts_df['game_start_date'] = hosts_df['game_start_date'].apply(lambda x: pd.to_datetime(x))

# Lets check how many days each season lasted

hosts_df['season_days'] = (hosts_df['game_end_date'] - hosts_df['game_start_date'])
hosts_df['game_year'] = hosts_df['game_year'].astype(int)

# Yearwise Rank position
print("Yearwise Rank position: ", results_df.groupby(['game_year', 'rank_position']).count())

print("Unique values of Rank position: ", results_df[
    results_df['rank_position'] == results_df[results_df['game_year'] == '1896']['rank_position'].unique()[12]])

faulty_rank_positions = results_df[
    results_df['rank_position'] == results_df[results_df['game_year'] == '1896']['rank_position'].unique()[12]]
print("faulty_rank_positions: ", faulty_rank_positions.index)
results_df['rank_position'].iloc[faulty_rank_positions.index[0]] = results_df['rank_position'].mode()[0]

for index in range(len(faulty_rank_positions.index)):
    results_df.loc[faulty_rank_positions.index, 'rank_position'] = results_df['rank_position'].mode()[0]

# Yearwise Rank position
print("Yearwise Rank position after clean up: ", results_df.groupby(['game_year', 'rank_position']).count())

# Convert rank_position column to numeric type
results_df['rank_position'] = pd.to_numeric(results_df['rank_position'], errors='coerce')

# Drop the rows containing non-numeric values
results_df = results_df.dropna(subset=['rank_position'])
results_df['rank_position'].astype(str).str.slice(stop=1).astype(int)

# -------------------------------------------------------------------------------------------------------
# ---------------------------------------Cleaning Datasets Done------------------------------------------
# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
# -----------------------------------------Aggregation----------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# ---------------------------------------------athelete-----------------------------------------------------
# Average Participant's age per season
mean_age_by_first_year = athletes_df.groupby('first_game_year').mean(numeric_only=True)['athletes_age']
plt.figure()
plt.plot(mean_age_by_first_year.index, mean_age_by_first_year.values, marker='o')
plt.xlabel('First Game Year')
plt.ylabel('Mean Athlete Age')
plt.title('Mean Age of Athletes by First Game Year')
plt.show()

# Relation between age and no. of game participated
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.regplot(x="athletes_age", y="games_participations", data=athletes_df, scatter_kws={'s': 20})
plt.xlabel("Age of Athletes")
plt.ylabel("Number of Games Participated")
plt.title("Relation between Age and Number of Games Participated")
plt.show()

# -----------------------------------------hosts----------------------------------------
# top 5 countries who hosted multiple times
plt.figure()
top_locations = hosts_df['game_location'].value_counts().head(5)
plt.bar(top_locations.index, top_locations.values)
plt.xlabel('Game location')
plt.ylabel('Number of games hosted')
plt.title('Top 5 game locations')
plt.show()

hosts_df['season_days'] = hosts_df['season_days'].apply(lambda x: x.days)

# checking for outliers
plt.figure()
sns.displot(x=hosts_df['season_days'], bins=50)
plt.show()
# lets check the outliers

plt.figure()
sns.displot(x=hosts_df['season_days'], bins=50)
plt.ylim(0, 5)
plt.show()

# print(hosts_df[hosts_df['season_days'] > 75])
plt.figure()
sns.scatterplot(x=hosts_df['game_year'], y=hosts_df['season_days'])
plt.show()

# plotting game season and number of games taken in each season
plt.figure(figsize=(8, 6))
sns.barplot(x=hosts_df['game_season'].value_counts().index, y=hosts_df['game_season'].value_counts())
plt.title("Number of games taken in each season")
plt.xlabel("Game Season")
plt.ylabel("Number of Games")
plt.show()

# ------------------------Medals------------------------------------------------------------------

# top 5 countries with gold medals?

top_5_gold = medals_df[medals_df['medal_type'] == 'GOLD']['country_name'].value_counts().head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_gold.index, y=top_5_gold.values, palette="rocket")
plt.title('Top 5 countries with gold medals')
plt.xlabel('Country')
plt.ylabel('Number of gold medals')
plt.show()

# in which (top 10) sport USA has won most golds?

# Extract the data
usa_gold_sports = medals_df[(medals_df['medal_type'] == 'GOLD') &
                            (medals_df['country_name'] == 'United States of America')][
    'discipline_title'].value_counts().head(10)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x=usa_gold_sports.values, y=usa_gold_sports.index, palette='Blues_r')

# Add labels and title
plt.xlabel('Number of gold medals')
plt.ylabel('Sport discipline')
plt.title('Top 10 sport disciplines in which USA has won the most gold medals')

# Display the plot
plt.show()

# USA Gold medals per year

plt.figure(figsize=(25, 7), dpi=121)
sns.histplot(
    medals_df[(medals_df['medal_type'] == 'GOLD') & (medals_df['country_name'] == 'United States of America')][
        'game_year'], kde=True)
plt.title("USA Gold medals per year")
plt.tight_layout()
plt.show()

plt.figure(figsize=(25, 15), dpi=121)
sns.histplot(x=medals_df['game_year'], hue=medals_df['medal_type'])
plt.title("Number of medals of each type per year")
plt.tight_layout()
plt.show()

# In Which (Top10) games it was hard to win gold medal

print("In Which (Top10) games it was hard to win gold medal: \n",
      medals_df[medals_df['medal_type'] != 'GOLD']['discipline_title'].value_counts().head(10))

# --------------------------------------Results------------------------------------------------------------------

# Distribution of ranks of USA over the year
plt.figure(figsize=(25, 15))
sns.histplot(x=results_df[results_df[
                              'country_name'] == 'United States of America']['game_year'],
             y=results_df[results_df['country_name'] == 'United Statesof America']['rank_position'])
plt.title("Distribution of ranks of USA over the year")
plt.show()
# -------------------------------------------------------------------------------------------------------
# -----------------------------------------Aggregation Done---------------------------------------------------
# -------------------------------------------------------------------------------------------------------


# dropping irrelevant data features

# ----------------------------------------Merging dataset------------------------------------------------

data = medals_df.merge(hosts_df, how='left', left_on='slug_game', right_on='game_slug')
data2 = athletes_df.merge(results_df, how='left', left_on='athlete_full_name', right_on='athlete_full_name')
merged_data = data.merge(data2, on=['athlete_full_name', 'discipline_title'], how='outer')

merged_data.drop(['participant_type_x', 'country_name_x', 'game_location_x', 'game_end_date', 'game_start_date'],
                 axis=1, inplace=True)

# Drop rows with missing values for columns with missing value percentage between 10-15%
merged_data.dropna(
    subset=['event_title_y', 'slug_game_y', 'participant_type_y', 'rank_position', 'country_name_y', 'athlete_url_y'],
    inplace=True)
# Drop rows with missing values for columns with missing value percentage of around 5%
merged_data.dropna(
    subset=['athlete_url_x', 'games_participations', 'first_game', 'athlete_year_birth', 'first_game_year',
            'athletes_age'], inplace=True)
# Drop rows with missing values for column with missing value percentage less than 1%
merged_data.dropna(subset=['discipline_title', 'game_slug', 'game_year_x'], inplace=True)

merged_data.drop(
    ['athlete_full_name', 'game_slug', 'event_title_x', 'slug_game_x', 'first_game', 'athlete_url_y', 'athlete_url_x',
     'country_code', 'game_year_y',
     'country_3_letter_code_x', 'slug_game_y', 'game_year_x', 'country_3_letter_code_y'], axis=1,
    inplace=True)
merged_data[
    ['season_days', 'games_participations', 'athlete_year_birth', 'first_game_year', 'athletes_age', 'rank_position',
     'game_year']] = \
    merged_data[['season_days', 'games_participations', 'athlete_year_birth', 'first_game_year', 'athletes_age',
                 'rank_position', 'game_year']].astype(int)

print(merged_data.isnull().sum())

# -------------------------------------------------------------------------------------------------------
# -------------------------------------Dimensionality reduction/feature selection:-----------------------
# -------------------------------------------------------------------------------------------------------

# One-hot encode categorical columns
num_cols = ['season_days', 'games_participations', 'athlete_year_birth',
            'first_game_year', 'athletes_age', 'rank_position']
categorical_cols = ['discipline_title', 'event_gender', 'medal_type',
                    'game_location_y', 'game_name', 'game_season', 'event_title_y', 'participant_type_y',
                    'country_name_y', 'game_year']
encoded_data = pd.get_dummies(merged_data, columns=categorical_cols)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop('rank_position', axis=1),
                                                    encoded_data['rank_position'],
                                                    test_size=0.2, random_state=123)

# Fit Random Forest classifier on training data
rf = RandomForestClassifier(max_depth=16, random_state=123)
rf.fit(X_train, y_train)

# get the feature importances
feature_importances = rf.feature_importances_

# create a dataframe with the feature importances
feature_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})

# sort the dataframe by feature importance
feature_df = feature_df.sort_values('importance', ascending=False)

# keep only the top n features
num = 12
top_features = feature_df.head(num)['feature']

# select only the top features from your training and test data
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

print("top_features", top_features)
plt.figure()
plt.barh(feature_df['feature'][:16], feature_df['importance'][:16])
plt.title('Top 16 Features')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
#
# Encode the categorical columns using one-hot encoding
numerical_cols = ['season_days', 'games_participations', 'athlete_year_birth', 'first_game_year', 'athletes_age',
                  'rank_position', 'game_year']

categorical_cols = ['discipline_title', 'event_gender', 'medal_type', 'game_location_y', 'game_name', 'game_season',
                    'event_title_y', 'participant_type_y', 'country_name_y']

# perform SVD with n_components=16
n_components = 16
svd = TruncatedSVD(n_components=n_components)
X_train_svd = svd.fit_transform(X_train)

# calculate explained variance ratio for each component
explained_variance_ratio = svd.explained_variance_ratio_

# drop features based on explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
threshold = 0.95
drop_cols = []
for i, ratio in enumerate(cumulative_variance_ratio):
    if ratio > threshold:
        break
    for j, col in enumerate(X_train.columns):
        if col in numerical_cols:
            continue
        if svd.components_[i, j] != 0 and col not in drop_cols:
            drop_cols.append(col)

X_train_svd = X_train.drop(drop_cols, axis=1)
X_test_svd = X_test.drop(drop_cols, axis=1)

# PCA

n = 16
# Apply PCA
pca = PCA(n_components=n)
pca.fit(X_train)

# get the explained variance ratios for each principal component
explained_variances = pca.explained_variance_ratio_
# create a dataframe with the explained variance ratios
variance_df = pd.DataFrame({'component': range(1, n + 1), 'variance': explained_variances})

# sort the dataframe by variance
variance_df = variance_df.sort_values('variance', ascending=False)

# keep only the top m components that explain the most variance
m = 12
top_components = pca.components_[:m]

# Plot the cumulative explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

cumulative_variance_ratio = np.cumsum(explained_variances)
# set the threshold for variance explained
threshold = 0.975

# get the number of PCA components to keep based on the threshold
num_components_to_keep = np.argmax(cumulative_variance_ratio >= threshold) + 1

# get the list of features to drop
features_to_drop = merged_data.columns[len(num_cols):][:len(num_cols) - num_components_to_keep].tolist()

# drop the features with low variance
merged_data_pca = merged_data.drop(features_to_drop, axis=1)
print("merged data after pca is ", merged_data_pca)

# Load the data
# Compute the sample covariance matrix and display as heatmap
cov_matrix = merged_data.cov(numeric_only=True)
plt.figure()
sns.heatmap(cov_matrix, cmap="YlGnBu")
plt.title("Sample Covariance Matrix Heatmap")
plt.show()

# Compute the sample Pearson correlation coefficients matrix and display as heatmap
corr_matrix = merged_data.corr(numeric_only=True)
plt.figure()
sns.heatmap(corr_matrix, cmap="YlGnBu")
plt.title("Sample Pearson Correlation Coefficients Heatmap")
plt.show()
#
# ----------------------------------------Phase II: Regression Analysis----------------------------------------
# -------------------------------------------------------------------------------------------------------------
# Define the bins and labels for the rank positions
bins = [0, np.percentile(merged_data_pca['rank_position'], 10), np.percentile(merged_data_pca['rank_position'], 25),
        np.percentile(merged_data_pca['rank_position'], 50), np.percentile(merged_data_pca['rank_position'], 75), 101]
labels = ['top 10%', '10-25%', '25-50%', '50-75%', 'bottom 25%']

# Create a new column for binned rank positions
merged_data_pca['rank_position_bin'] = pd.cut(merged_data_pca['rank_position'], bins=bins, labels=labels)

# Perform a t-test between male and female athletes
male_ranks = merged_data_pca.loc[merged_data['event_gender'] == 'Men', 'rank_position']
female_ranks = merged_data_pca.loc[merged_data['event_gender'] == 'Women', 'rank_position']
t_statistic, p_value = ttest_ind(male_ranks, female_ranks, equal_var=False)
print('Men vs Women t-test results: t-statistic = {}, p-value = {}'.format(t_statistic, p_value))

# Perform a t-test between athletes from two different countries
country1_ranks = merged_data_pca.loc[merged_data_pca['country_name_y'] == 'Canada', 'rank_position']
country2_ranks = merged_data_pca.loc[merged_data_pca['country_name_y'] == 'United States of America', 'rank_position']
t_statistic, p_value = ttest_ind(country1_ranks, country2_ranks, equal_var=False)
print('Canada vs United States of America t-test results: t-statistic = {}, p-value = {}'.format(t_statistic, p_value))

# Association analysis :

# Perform correlation analysis between the top 5 features

plt.figure()
corr_matrix = merged_data_pca.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Association Analysis between features")
plt.show()

# f test analysis :

# Get the rank position data for each bin
top_10 = merged_data_pca[merged_data_pca['rank_position_bin'] == 'top 10%']['rank_position']
ten_to_25 = merged_data_pca[merged_data_pca['rank_position_bin'] == '10-25%']['rank_position']
twofive_to_50 = merged_data_pca[merged_data_pca['rank_position_bin'] == '25-50%']['rank_position']
fifty_to_75 = merged_data_pca[merged_data_pca['rank_position_bin'] == '50-75%']['rank_position']
bottom_25 = merged_data_pca[merged_data_pca['rank_position_bin'] == 'bottom 25%']['rank_position']

# Perform the F-test
f_stat, p_value = f_oneway(top_10, ten_to_25, twofive_to_50, fifty_to_75, bottom_25)

# Print the results
print("F-statistic:", f_stat)
print("p-value:", p_value)

# Scaling data

X = merged_data_pca.drop(columns=['rank_position'])
y = merged_data_pca['rank_position']

num_cols = ['athletes_age', 'game_year']
categorical_cols = ['discipline_title', 'event_gender', 'medal_type',
                    'game_location_y', 'game_name', 'game_season', 'event_title_y', 'participant_type_y',
                    'country_name_y', 'rank_position_bin']
encoded_data1 = pd.get_dummies(X, columns=categorical_cols)

scaled_data = (X[num_cols] - X[num_cols].mean()) / X[num_cols].std()
scaled_data = pd.DataFrame(scaled_data, columns=num_cols)
merged_scaled_data = pd.concat([scaled_data, encoded_data1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(merged_scaled_data,
                                                    y, test_size=0.2, shuffle=True,
                                                    random_state=123)
# regression model
X = sm.add_constant(X_train)
included_features = list(X_train.columns)
model = sm.OLS(y_train, sm.add_constant(X_train[included_features])).fit()
y_pred = model.predict(X_test[included_features]).round()

print(model.summary())
# Calculate the confidence intervals
conf_int = model.conf_int()
conf_int['OR'] = model.params
conf_int.columns = ['Lower CI', 'Upper CI', 'OR']
conf_int.to_excel('confidence_intervals.xlsx')
print("confidence interval :\n", np.exp(conf_int))

#  Stepwise regression and adjusted R-square analysis.

p_values = model.pvalues[1:]
print("p values greater than 0.05", p_values[p_values > 0.05].index.tolist())

adj_r = model.rsquared_adj
print('Adjusted R-square Initially before variable elimination: ', round(adj_r, 3))

X2 = sm.add_constant(X_train)
included_features2 = p_values[p_values < 0.06].index.tolist()

model2 = sm.OLS(y_train, sm.add_constant(X_train[included_features2])).fit()
y_predmodel2 = model2.predict(X_test[included_features2]).round()

adj_r22 = model2.rsquared_adj
print('Adjusted R-square after remodelling and removal of features:  ', round(adj_r22, 3))

y_predmodel1 = model.predict(X_test[included_features]).round()
accuracy = accuracy_score(y_test, y_predmodel1)
print("Accuracy of the model before variable elimination: ", round(accuracy, 3))
accuracy2 = accuracy_score(y_test, y_predmodel2)
print("Accuracy of the model after variable elimination: ", round(accuracy2, 3))


# ---------------------------------------Phase III: Classification Analysis:-----------------------------------
# -------------------------------------------------------------------------------------------------------------

X_train1, X_test1 = X_train[included_features2], X_test[included_features2]

# train the decision tree classifier
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_train, y_train)

# make predictions on the testing set
y_pred1 = dt.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# evaluate the model performance
accuracy = accuracy_score(y_test, y_pred1)
dt_cm = confusion_matrix(y_test, y_pred1)
print('Decision Tree Accuracy:', round(accuracy,3))
print('Decision Tree Confusion Matrix:', dt_cm)


# Display Precision
precision = metrics.precision_score(y_test, y_pred1,average='weighted')
print('Decision Tree Precision:', round(precision,3))

# Display Sensitivity or Recall
recall = metrics.recall_score(y_test, y_pred1,average='weighted')
print('Decision Tree Sensitivity/Recall:', round(recall,3))

# Display Specificity
cm = confusion_matrix(y_test, y_pred1, labels=np.unique(y_test))
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
specificity = tn / (tn + fp)
print('Decision Tree Specificity:', round(specificity,3))

# Display F-score
f1 = f1_score(y_test, y_pred1,average='weighted')
print('Decision Tree F-score:', round(f1,3))

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred1)
print(report)
# Calculate and plot ROC curve
y_prob = dt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Decision Tree')
plt.legend(loc="lower right")
plt.show()

# KNN
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
knn.fit(X_train1, y_train)
y_pred2 = knn.predict(X_test1)
knn_acc = accuracy_score(y_test, y_pred2)
knn_cm = confusion_matrix(y_test, y_pred2)
print('KNN Accuracy:', round(knn_acc,3))

# Display Precision
knn_precision = precision_score(y_test, y_pred2, average='macro')
print('KNN Precision:', round(knn_precision,3))

# Display Sensitivity or Recall
knn_recall = recall_score(y_test, y_pred2, average='macro')
print('KNN Recall:', round(knn_recall,3))

# Display Specificity
cm1 = confusion_matrix(y_test, y_pred2, labels=np.unique(y_test))
tn1, fp1, fn1, tp1 = cm1[0,0], cm1[0,1], cm1[1,0], cm1[1,1]
specificity = tn1 / (tn1 + fp1)
print('KNN Specificity:', round(specificity,3))


# Display F-score
knn_fscore = f1_score(y_test, y_pred2, average='macro')
print('KNN F-score:', round(knn_fscore,3))

# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred2)
print(matrix)

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred2)
print(report)

# Calculate and plot ROC curve
y_prob = knn.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for KNN')
plt.legend(loc="lower right")
plt.show()



# Random Forest
rf = RandomForestClassifier(random_state=123)
rf.fit(X_train1, y_train)
y_pred3 = rf.predict(X_test1)

rf_acc = accuracy_score(y_test, y_pred3)
rf_cm = confusion_matrix(y_test, y_pred3)
rf_report = classification_report(y_test, y_pred3)

# Display metrics
print('Random Forest Accuracy:', round(rf_acc,3))


# Display Precision
rff_precision = precision_score(y_test, y_pred3, average='macro')
print('Random Forest Precision:', round(rff_precision,3))

# Display Sensitivity or Recall
rff_recall = recall_score(y_test, y_pred3, average='macro')
print('Random Forest Recall:', round(rff_recall,3))

# Display Specificity
cm2 = confusion_matrix(y_test, y_pred3, labels=np.unique(y_test))
tn2, fp2, fn2, tp2 = cm2[0,0], cm2[0,1], cm2[1,0], cm2[1,1]

precision = tp2 / (tp2 + fp2)
sensitivity = tp2 / (tp2 + fn2)
specificity = tn2 / (tn2 + fp2)
f_score = 2 * precision * sensitivity / (precision + sensitivity)
print('Random Forest Specificity:', round(specificity,3))
print('Random Forest F-score:', round(f_score,3))

print('Random Forest Confusion Matrix:', rf_cm,3)
print('Random Forest Classification Report:', rf_report,3)

# Calculate and plot ROC curve
y_prob = rf.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds1 = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Random forest')
plt.legend(loc="lower right")
plt.show()

#Naive Bayes

nb = GaussianNB()
nb.fit(X_train1, y_train)
y_pred4 = nb.predict(X_test1)
nb_acc = accuracy_score(y_test, y_pred4)
nb_cm = confusion_matrix(y_test, y_pred4)

print('Naïve Bayes Accuracy:', round(nb_acc,3))

nb_precision = precision_score(y_test, y_pred4, average='macro')
print('Naïve Bayes Precision:', round(nb_precision,3))

nb_recall = recall_score(y_test, y_pred4, average='macro')
print('Naïve Bayes Recall:', round(nb_recall,3))

cm3 = confusion_matrix(y_test, y_pred4, labels=np.unique(y_test))
tn2, fp2, fn2, tp2 = cm2[0,0], cm2[0,1], cm2[1,0], cm2[1,1]
specificity = tn2 / (tn2 + fp2)
print('Naïve Bayes Specificity:', round(specificity,3))

nb_fscore = f1_score(y_test, y_pred4, average='macro')
print('Naïve Bayes F-score:', nb_fscore)

print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred4)
print(matrix)

print("\nClassification Report")
report = classification_report(y_test, y_pred4)
print(report)

accuracy = accuracy_score(y_test, y_pred4)
print('Naïve Bayes Classification Accuracy of the model: {:.2f}%'.format(accuracy * 100))
y_prob4 = nb.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds2 = roc_curve(y_test, y_prob4, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Naïve Bayes')
plt.legend(loc="lower right")
plt.show()



# Logistic Regression
logreg = LogisticRegression(solver='sag', random_state=123, max_iter=100)
logreg.fit(X_train1, y_train)
y_pred5 = logreg.predict(X_test1)
logreg_acc = accuracy_score(y_test, y_pred5)
print('Logistic Regression Accuracy:', round(logreg_acc,3))
logreg_precision = precision_score(y_test, y_pred5, average='macro')
print('Logistic Regression Precision:', round(logreg_precision,3))
logreg_recall = recall_score(y_test, y_pred5, average='macro')
print('Logistic Regression Recall:', round(logreg_recall,3))
cm4 = confusion_matrix(y_test, y_pred5, labels=np.unique(y_test))
tn3, fp3, fn3, tp3 = cm4[0,0], cm4[0,1], cm4[1,0], cm4[1,1]
specificity = tn3 / (tn3 + fp3)
print('Logistic Regression Specificity:', round(specificity,3))
logreg_fscore = f1_score(y_test, y_pred5, average='macro')
print('Logistic Regression F-score:', round(logreg_fscore,3))

print("Logistic Regression Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred5)
print(matrix)

print("Classification Report\n")
report = classification_report(y_test, y_pred5)
print(report)
y_prob5 = logreg.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds3 = roc_curve(y_test, y_prob5, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Logistic Regression')
plt.legend(loc="lower right")
plt.show()



#
#
# # SVM
# # train the SVM classifier
# svm = SVC(kernel='linear', random_state=123, probability=True)
# svm.fit(X_train1, y_train)
#
# # make predictions on the testing set
# y_pred7 = svm.predict(X_test)
#
# # evaluate the model performance
# accuracy = accuracy_score(y_test, y_pred7)
# svm_cm = confusion_matrix(y_test, y_pred7)
# print('SVM Accuracy:', round(accuracy, 3))
# print('SVM Confusion Matrix:', svm_cm)
#
# # Display Precision
# precision = metrics.precision_score(y_test, y_pred7, average='weighted')
# print('SVM Precision:', round(precision, 3))
#
# # Display Sensitivity or Recall
# recall = metrics.recall_score(y_test, y_pred7, average='weighted')
# print('SVM Sensitivity/Recall:', round(recall, 3))
#
# # Display Specificity
# cm6 = confusion_matrix(y_test, y_pred7, labels=np.unique(y_test))
# tn5, fp5, fn5, tp5 = cm6[0, 0], cm6[0, 1], cm6[1, 0], cm6[1, 1]
# specificity = tn5 / (tn5 + fp5)
# print('SVM Specificity:', round(specificity, 3))
#
# # Display F-score
# f1 = f1_score(y_test, y_pred7, average='weighted')
# print('SVM F-score:', round(f1, 3))
#
# # Calculate and plot ROC curve
# y_prob6 = svm.predict_proba(X_test1)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob6, pos_label=1)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic for SVM')
# plt.legend(loc="lower right")
# plt.show()

# Phase4: Clustering and association analysis

# ==========================================================
# Elbow Method unsupervised kmean method for clustering
# =========================================================

#N is used to set the maximum number of clusters to be evaluated using the elbow method
N = 10
wcss = []
for i in range(1, N):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X_train1)
    wcss.append(kmeans.inertia_)
plt.plot(np.arange(1, N, 1), np.array(wcss), lw=3)
plt.xticks(np.arange(1, N, 1))
plt.title('Optimum k in knn method Elbow method ')
plt.ylabel('wcss')
plt.xlabel('number of clusters')
plt.grid()
plt.tight_layout()
plt.show()

modelKN = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2)
modelKN.fit(X_train1, y_train)
# Predicting the results
y_pred_e = modelKN.predict(X_test1)
print(f"Accuracy with elbow method {100 * metrics.accuracy_score(y_test, y_pred_e):.3f}%")
print(f"Precision with elbow method{100 * metrics.precision_score(y_test, y_pred_e, average='micro'):.3f}%")
print(f"Recall with elbow method {100 * metrics.recall_score(y_test, y_pred_e, average='micro'):.3f}%")

cm7 = confusion_matrix(y_test, y_pred_e, labels=np.unique(y_test))
tn7, fp7, fn7, tp7 = cm7[0,0], cm7[0,1], cm7[1,0], cm7[1,1]
specificity = tn7 / (tn7 + fp7)
print('elbow method Specificity:', round(specificity,3))
knn_elbow_fscore = f1_score(y_test, y_pred_e, average='macro')
print('elbow method F-score:', round(knn_elbow_fscore,3))

print("elbow method Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred_e)
print(matrix)

print("elbow method Classification Report\n")
report = classification_report(y_test, y_pred_e)
print(report)
y_prob7 = modelKN.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds3 = roc_curve(y_test, y_prob7, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for KNN after K mean with elbow method')
plt.legend(loc="lower right")
plt.show()

#Apriori
# Convert the dataset into a binary format using one-hot encoding
data_binary = (merged_data_pca != 0)

# Apply the Apriori algorithm to find frequent itemsets with a minimum support of 0.05
frequent_itemsets = apriori(data_binary, min_support=0.05, use_colnames=True)

# Generate association rules with a minimum confidence of 0.8
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Print the frequent itemsets and association rules

top_frequent_itemsets = frequent_itemsets.sort_values(by=['support'], ascending=False)[:10]
top_rules = rules.sort_values(by=['confidence'], ascending=False)[:10]

print("frequent_features sets: \n",top_frequent_itemsets)
print("Association Rules :\n",top_rules)
