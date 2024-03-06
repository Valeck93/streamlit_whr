import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time

from scipy.stats import pearsonr
import statsmodels.api

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap

df_merge = pd.read_csv("df_merge.csv")
df_merge = df_merge.drop("Unnamed: 0", axis=1)
df_anova = df_merge.rename(columns={"Life Ladder": "Life_Ladder", "Regional indicator":"Regional_indicator", "Log GDP per capita":"Log_GDP_per_capita", "Social support":"Social_support", "Freedom to make life choices":"Freedom_to_make_life_choices"})
df_anova_cov = df_anova[(df_anova["pre_post"] != "none")]
df_corr = df_merge.dropna(axis=0)

dtypes = df_merge.dtypes.value_counts()
outliers = 0
extremes = "several"
num_feats0 = 8
cat_feats0 = 2

display_index = {0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
                   8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"}

feats = df_merge.drop(["Country name", "year", "Life Ladder"], axis=1)
target = df_merge["Life Ladder"]
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=5)
  
num_train = X_train[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]]
num_test = X_test[["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]]
cat_train = X_train[["Regional indicator", "pre_post"]]
cat_test = X_test[["Regional indicator", "pre_post"]]

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
num_train = imputer.fit_transform(num_train)
num_test = imputer.transform(num_test)
scaler = MinMaxScaler()
num_train = scaler.fit_transform(num_train)
num_test = scaler.transform(num_test)
ohe = OneHotEncoder(drop="first", sparse_output=False)
cat_train = ohe.fit_transform(cat_train)
cat_test = ohe.transform(cat_test)

X_train = pd.DataFrame(np.concatenate([num_train, cat_train], axis=1))
X_test = pd.DataFrame(np.concatenate((num_test, cat_test), axis=1))

st.markdown("<h2 style='text-align: center;'>World Happiness Report</h2>", unsafe_allow_html=True)
st.write("_____")
st.sidebar.title("Project Steps")
pages=["0. Cover", "1. Introduction", "2. Understanding Data", "3. Data Visualization", "4. Hypothesis Tests", "5. Modelling Pre-Processing", "6. First Wave Models", "7. Model Optimization", "8. Limitations & Outlook"]
page=st.sidebar.radio("Select:", pages)

if page == pages[0] : 
  col1, col2, col3 = st.columns(3)
  with col1:
    st.write(" ")
  with col2:
    st.image("WHR_01.png")
  with col3:
    st.write(" ")

  st.markdown("<h3 style='text-align: center;'>Presenting Data Exploration, Data Visualization and Modelling Process</h3>", unsafe_allow_html=True)
  st.markdown("<h5 style='text-align: center;'>by Valentin Eckhardt, Dimitrios Kastanis, Inji Mammadova</h5>", unsafe_allow_html=True)
  st.markdown("<h5 style='text-align: center;'>26th February 2024", unsafe_allow_html=True)

if page == pages[1] : 
  st.markdown("<h3 style='text-align: center;'>[ Introduction ]</h3>", unsafe_allow_html=True)
  st.subheader("What is the World Happiness Report?")
  st.write("Yearly report from Sustainable Development Solutions Network")
  st.write("- academic, institutional, educational, political and commercial partners")
  st.write("- development of international happiness measures")
  st.write("Based on Gallup World Poll survey data")
  st.write("- approx. 1000 interviewees per country, over 100 questions")
  st.write("- over 160 countries")
  st.write("- since 2005")

  st.subheader("What Data are we working with?")
  st.write("*world-happiness-report.csv*, summarizes 2005-2020")
  st.write("*world-happiness-report-2021.csv*, details 2021")
  st.write("At the core: 6 key factors")
  st.write("- :red[Logged GDP per capita]")
  st.write("- :red[Social support]")
  st.write("- :red[Healthy life expectancy at birth]")
  st.write("- :red[Freedom to make life choices]")
  st.write("- :red[Generosity]")
  st.write("- :red[Perception of corruption]")
  st.write("Additionally: 2 affect variables")
  st.write("- :red[Positive affect] (*happiness, laughing and enjoyment*)")
  st.write("- :red[Negative affect] (*worry, sadness and anger, respectively*)")

  st.subheader("What is our objective?")
  st.write("*Which factors influence happiness the most?*")
  st.write("*What combination of factors explains the difference in country’s happiness rankings best?*")

if page == pages[2] :
  st.markdown("<h3 style='text-align: center;'>[ Understanding Data ]</h3>", unsafe_allow_html=True)
  st.subheader("1. Checking")
  if st.checkbox("Data Types") :
    st.write(dtypes)
  if st.checkbox("Missing Values (in %)") : 
    st.dataframe(df_merge.isna().sum()/len(df_merge)*100, use_container_width=True)
  if st.checkbox("Distributions") :
    st.write("(Extremes:", "several","/", "Outliers:", outliers, ")")
  if st.checkbox("Suitability for ML") :
    st.write("(Numerical features:", num_feats0 , "/", "Categorical features:", cat_feats0, ")")
    st.write(target)

  st.subheader("2. Merging")
  st.write("WHR Summary and WHR 2021 :arrow_right: MERGE")
  st.write("Rows:", df_merge.shape[0], "Columns:", df_merge.shape[1])
  
  st.subheader("3. Enriching")
  st.write("New variable *pre/post Covid*")
  st.write("*Regional indicators* across whole dataset")
  st.write("Removal of *2005* because only 27 countries included")
  if st.checkbox("Show MERGE dataset") :
    st.dataframe(df_merge)

if page == pages[3] :
  st.markdown("<h3 style='text-align: center;'>[ Data Visualization ]</h3>", unsafe_allow_html=True)
  choice = ["Correlation Matrix", "Barplot: Ladder Score by Region", "Barplot: Ladder Score by Covid", "Scatterplot: Ladder Score & Log GPD", "Lineplot: Global Ladder Score evolution"]
  option = st.selectbox("Chosen Figure", options=choice, label_visibility="hidden")

  if option == "Correlation Matrix" :
    st.image("correlations_merge.png")
  if option == "Barplot: Ladder Score by Region" :
    st.image("life_ladder_region.png") 
  if option == "Barplot: Ladder Score by Covid" :
    st.image("life_ladder_regio_covid.png")
  if option == "Scatterplot: Ladder Score & Log GPD" :
    st.image("life_ladder_gdp_regio.png")
  if option == "Lineplot: Global Ladder Score evolution" :
    st.image("life_ladder_years.png")

if page == pages[4] :
  st.markdown("<h3 style='text-align: center;'>[ Hypothesis Tests ]</h3>", unsafe_allow_html=True)
  
  result1 = statsmodels.formula.api.ols('Life_Ladder ~ Regional_indicator', data=df_anova).fit()
  table1 = statsmodels.api.stats.anova_lm(result1)
  result2 = statsmodels.formula.api.ols('Life_Ladder ~ pre_post', data=df_anova_cov).fit()
  table2 = statsmodels.api.stats.anova_lm(result2)
  result3 = statsmodels.formula.api.ols('Log_GDP_per_capita ~ Regional_indicator', data=df_anova).fit()
  table3 = statsmodels.api.stats.anova_lm(result3)
  result4 = statsmodels.formula.api.ols('Social_support ~ pre_post', data=df_anova_cov).fit()
  table4 = statsmodels.api.stats.anova_lm(result4)
  prs_r = pearsonr(x = df_corr["Life Ladder"], y = df_corr["Generosity"])[1]
  prs_p = pearsonr(x = df_corr["Life Ladder"], y = df_corr["Generosity"])[0]

  st.subheader("*:heavy_check_mark: 1. Regional indicator does influence Life Ladder*")
  st.write("[ANOVA]", "p-Value: <", table1.iloc[0][-1])
  st.subheader(":heavy_check_mark: *2. Pre/Post Covid does influence Life Ladder*") 
  st.write("[ANOVA]", "p-Value:", table2.iloc[0][-1])
  st.subheader(":heavy_check_mark: *3. Regional indicator does influence Log GDP*")
  st.write("[ANOVA]", "p-Value: <", table3.iloc[0][-1])
  st.subheader(":x: *4. Pre/Post Covid does not influence Social support*")
  st.write("[ANOVA]", "p-Value:", table4.iloc[0][-1])
  st.subheader(":heavy_check_mark: *5. Generosity is correlated with Life Ladder*")
  st.write("[Pearson-Correlation-Test]", "p-Value: <", round(prs_r,4), "Coefficient:",round(prs_p,4))

if page == pages[5] :
  st.markdown("<h3 style='text-align: center;'>[ Modelling Pre-Processing ]</h3>", unsafe_allow_html=True)
  test_size0 = 0.25

  st.subheader("1. Dividing Data into Training and Test Set")
  st.write("with *train_test_split*() @ *testsize =*", test_size0)
  st.subheader("2. Normalization")
  st.write("with *MinMaxScaler*()")
  st.subheader("3. Imputing")
  st.write("with *SimpleImputer*() @ *strategy = mean*")
  st.subheader("4. Encoding")
  st.write("with *OneHotEncoder*()")

  if st.checkbox("Show X_train as ML-ready Input") :
    st.dataframe(X_train)

if page == pages[6] :
  st.markdown("<h3 style='text-align: center;'>[ First Wave Models ]</h3>", unsafe_allow_html=True)
  
  my_bar = st.progress(0)
  for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1)
  my_bar.empty()

  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
  y_pred_lr = regressor.predict(X_test)
  r2train_01 = regressor.score(X_train, y_train)
  r2test_01 = regressor.score(X_test, y_test)
  mse_01 = mean_squared_error(y_test, y_pred_lr)
  rmse_01 = np.sqrt(mse_01)
  mae_01 = mean_absolute_error(y_test, y_pred_lr)
  
  dt_regressor = DecisionTreeRegressor(random_state=42)
  dt_regressor.fit(X_train, y_train)
  y_pred_dt = dt_regressor.predict(X_test)
  r2train_02 = dt_regressor.score(X_train, y_train)
  r2test_02 = dt_regressor.score(X_test, y_test)
  mse_02 = mean_squared_error(y_test, y_pred_dt)
  rmse_02 = np.sqrt(mse_02)
  mae_02 = mean_absolute_error(y_test, y_pred_dt)

  rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42)
  rf_regressor.fit(X_train, y_train)
  y_pred_rf = rf_regressor.predict(X_test)
  r2train_03 = rf_regressor.score(X_train, y_train)
  r2test_03 = rf_regressor.score(X_test, y_test)
  mse_03 = mean_squared_error(y_test, y_pred_rf)
  rmse_03 = np.sqrt(mse_03)
  mae_03 = mean_absolute_error(y_test, y_pred_rf)

  xgb_regressor = XGBRegressor(random_state=42)
  xgb_regressor.fit(X_train, y_train)
  y_pred_xgb = xgb_regressor.predict(X_test)
  r2train_04 = xgb_regressor.score(X_train, y_train)
  r2test_04 = xgb_regressor.score(X_test, y_test)
  mse_04 = mean_squared_error(y_test, y_pred_xgb)
  rmse_04 = np.sqrt(mse_04)
  mae_04 = mean_absolute_error(y_test, y_pred_xgb)

  metrics_first = pd.DataFrame(data={"Linear Regression": [r2train_01, r2test_01, mse_01, rmse_01, mae_01], "Decision Tree": [r2train_02, r2test_02, mse_02, rmse_02, mae_02], 
                                     "Random Forest": [r2train_03, r2test_03, mse_03, rmse_03, mae_03], "XGBoost": [r2train_04, r2test_04, mse_04, rmse_04, mae_04]}, 
                                     index=["R² train", "R² test", "MSE", "RMSE", "MAE"])
  st.write(" ")
  st.subheader("Performance Metrics")
  st.dataframe(metrics_first, use_container_width=True)
  
  coeffs = list(regressor.coef_)
  coeffs.insert(0, regressor.intercept_)
  reg_feats = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect",
             "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"]
  reg_coefficients = pd.DataFrame({"Coefficient": coeffs}, index=reg_feats)
  reg_coefficients = reg_coefficients.iloc[(-reg_coefficients.Coefficient.abs()).argsort()]

  dt_feat_importances = pd.DataFrame(dt_regressor.feature_importances_, columns=["Importance"])
  dt_feat_importances.rename(index=display_index, inplace=True)
  dt_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)

  rf_feat_importances = pd.DataFrame(rf_regressor.feature_importances_, columns=["Importance"])
  rf_feat_importances.rename(index=display_index, inplace=True)
  rf_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)

  xgb_feat_importances = pd.DataFrame(xgb_regressor.feature_importances_, columns=["Importance"])
  xgb_feat_importances.rename(index=display_index, inplace=True)
  xgb_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)

  st.write(" ")
  st.subheader("Coefficients & Feature Importances")
  col4, col5 = st.columns(2)
  with col4:
    st.write("Linear Regression")
    st.dataframe(reg_coefficients, use_container_width=True)
  with col5:
    st.write("Decision Tree")
    st.dataframe(dt_feat_importances, use_container_width=True)

  col6, col7 = st.columns(2)
  with col6:
    st.write("Random Forest")
    st.dataframe(rf_feat_importances, use_container_width=True)
  with col7:
    st.write("XGBoost")
    st.dataframe(xgb_feat_importances, use_container_width=True)
    
if page == pages[7] :
  st.markdown("<h3 style='text-align: center;'>[ Model Optimization ]</h3>", unsafe_allow_html=True)
  st.subheader("Linear Regression: SHAP scores")
  st.image("shap_values_lr_01.png")
  st.write(" ")

  st.subheader("Linear Regression Optimization")
  st.write("- 2. Attempt: Coefficient Threshold 0.1")
  st.write("*Removal of pre/post, regional indicators: CIS, Middle East*")
  st.write("- 3. Attempt: De-Enrichment")
  st.write("*Removal of pre/post and all regional indicators*")
  st.write("- 4. Attempt: Coefficient Threshold 1.0")
  st.write("*Removal of pre/post, all regional indicators and Freedom, Generosity, Corruption, Negative affect*")
  st.write("___")

  st.subheader("Random Forest: SHAP scores")
  st.image("shap_values_rf_01.png")
  st.write(" ")
  st.subheader("Random Forest Optimization")
  st.write("- 2. Attempt: Parameter optimization with GridSearchCV")
  st.write("*n_estimators=80, max_depth=7, min_sample_split=2*")
  st.write("- 3. Attempt: SHAP value Threshold 0.05")
  st.write("*Removal of pre/post and regional indicators: CIS, Middle East, North America, Southeast Asia, Sub-Saharan Africa, Western Europe*")
  st.write("- 4. Attempt: De-Enrichment")
  st.write("*Removal of pre/post and all regional indicators*")
  st.write("___")

  X_train_lr = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
  X_train_lr = X_train_lr.drop(["Freedom to make life choices", "Generosity", "Perceptions of corruption", "Negative affect", "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", 
                            "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

  X_test_lr = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
  X_test_lr = X_test_lr.drop(["Freedom to make life choices", "Generosity", "Perceptions of corruption", "Negative affect", "Commonwealth of Independent States" , "East Asia", "Latin America and Caribbean", "Middle East and North Africa", "North America and ANZ", 
                          "South Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

  regressor_final = LinearRegression()
  regressor_final.fit(X_train_lr, y_train)
  y_pred_lr_final = regressor_final.predict(X_test_lr)
  r2train_lr_final = regressor_final.score(X_train_lr, y_train)
  r2test_lr_final = regressor_final.score(X_test_lr, y_test)
  mse_lr_final = mean_squared_error(y_test, y_pred_lr_final)
  rmse_lr_final = np.sqrt(mse_lr_final)
  mae_lr_final = mean_absolute_error(y_test, y_pred_lr_final)

  lr_final_coeffs = list(regressor_final.coef_)
  lr_final_coeffs.insert(0, regressor_final.intercept_)
  lr_final_feats = ["Intercept", "Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Positive affect"]
  lr_final_coefficients = pd.DataFrame({"Estimated value": lr_final_coeffs}, index=lr_final_feats)

  X_train_rf = pd.DataFrame(X_train).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
  X_train_rf = X_train_rf.drop(["Commonwealth of Independent States" , "Middle East and North Africa", "North America and ANZ", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)
  X_test_rf = pd.DataFrame(X_test).rename(columns={0:"Log GDP per capita", 1:"Social support", 2:"Healthy life expectancy at birth", 3:"Freedom to make life choices", 4:"Generosity", 5:"Perceptions of corruption", 6:"Positive affect", 7:"Negative affect",
             8:"Commonwealth of Independent States" , 9:"East Asia", 10:"Latin America and Caribbean", 11:"Middle East and North Africa", 12:"North America and ANZ", 13:"South Asia", 14:"Southeast Asia", 15:"Sub-Saharan Africa", 16:"Western Europe", 17:"post", 18:"pre"})
  X_test_rf = X_test_rf.drop(["Commonwealth of Independent States" , "Middle East and North Africa", "North America and ANZ", "Southeast Asia", "Sub-Saharan Africa", "Western Europe", "post", "pre"], axis=1)

  rf_regressor_final = RandomForestRegressor(
    n_estimators=80,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=5,
    random_state=42)
  rf_regressor_final.fit(X_train_rf, y_train)
  y_pred_rf_final = rf_regressor_final.predict(X_test_rf)
  r2train_rf_final = rf_regressor_final.score(X_train_rf, y_train)
  r2test_rf_final = rf_regressor_final.score(X_test_rf, y_test)
  mse_rf_final = mean_squared_error(y_test, y_pred_rf_final)
  rmse_rf_final = np.sqrt(mse_rf_final)
  mae_rf_final = mean_absolute_error(y_test, y_pred_rf_final)

  rf_final_feat_importances = pd.DataFrame(rf_regressor_final.feature_importances_, columns=["Importance"])
  rf_final_feat_importances.rename(index=display_index, inplace=True)
  rf_final_feat_importances.sort_values(by="Importance", ascending=False, inplace=True)

  metrics_final = pd.DataFrame(data={"Linear Regression [4th]": [r2train_lr_final, r2test_lr_final, mse_lr_final, rmse_lr_final, mae_lr_final, 4], "Random Forest [3rd]": [r2train_rf_final, r2test_rf_final, mse_rf_final, rmse_rf_final, mae_rf_final, 11]}, 
                               index=["R² train", "R² test", "MSE", "RMSE", "MAE", "No. Features"])
  
  st.subheader("Final Model Metrics")
  st.dataframe(metrics_final, use_container_width=True)
  st.write("The :red[Linear Regression model] – while not the highest performing – still is able to explain 75% of variance based only on four variables. Life Ladder scores can be explained with only Log GDP, Social support, Healthy life expectancy and Positive affect by a substantial part. In other words: if we want to improve Life Ladder scores for any given country, the most effective approach would be to focus on increasing these four measures.")
  st.write("For best possible predictions of Life Ladder scores with the given data the :red[Random Forest model] is the most suitable choice, especially if the complexity of the model does not play a major role. Since it does explain 85% of variance, with an MAE of only", round(metrics_final.iloc[4][1],4), ", we can expect the predictions made with this model to be quite good with only relatively small errors on average.")

if page == pages[8] :
  st.markdown("<h3 style='text-align: center;'>[ Limitations & Outlook ]</h3>", unsafe_allow_html=True)
  st.subheader("About Data Limitations")
  st.write("- :red[Based on survey data]: hard to control for external influences")
  st.write("- :red[Six chosen key factors might inherit bias]: cultural differences on what defines happiness")
  st.write("- :red[Years differ greatly in coverage of countries]: 89-149 per year, some countries appearing only once!")
  st.subheader("Time for Improvement")
  st.write("- Greater repertoire of advanced algorithms, optimization and interpretation techniques")
  st.write("- :red[XGBoost looked promising]: lack of time, experience and strikingly differing results made us drop it")
  st.write("- :red[Follow-Up project could include 2022 & 2023]: extended pre/post Covid period to analyze")
  st.write("- :red[Modellings for each Region]: test transferability of results")
  st.write("- :red[Compare countries that are economically comparable]: reduce Log GPD's importance")

