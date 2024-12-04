# Recipe Ratings Prediction

This project aims to predict recipe ratings using data from Food.com. The learning pipeline follows a standard machine learning framework, beginning with comprehensive data cleaning and exploratory data analysis. The project frames a prediction problem focused on predicting recipe ratings and systematically enhances model performance through development of optimized model. 

The project, authored by Fang Yu (fangyua@umich.edu), was conducted at University of Michigan.

---

## Introduction

The dataset, sourced from Food.com, contains recipes and ratings posted since 2008. It includes detailed information about recipes, such as preparation times, nutrition information, and user ratings. The focus of this project is to predict recipe ratings and explore the question: what types of recipes tend to receive higher ratings? I intend to utilize data analysis techniques (e.g. pandas and numpy) to evaluate the relationship between ratings and various columns. Based on the information derived from exploratory data analysis, I can construct a prediction model to predict the ratings of recipes. Specifically, recipe rating is a crucial factor for users to decide which recipe to follow. These insights can help users select recipes that highly align with their preferences. With a total of 234,429 rows, this dataset becomes a perfect choice for constructing such a prediction model. Here is a brief descriptions to each of the relevant columns:

| **Column**         | **Description**                        |
|---------------------|----------------------------------------|
| `name`           | Recipe name                           |
| `id`             | Recipe ID                             |
| `minutes`        | Minutes to prepare the recipe         |
| `submitted`      | Date the recipe was submitted         |
| `tags`           | Food.com tags for the recipe          |
| `n_steps`        | Number of steps in the recipe         |
| `ingredients`    | List of ingredients used in the recipe|
| `n_ingredients`  | Number of ingredients in the recipe   |
| `rating_avg`     | Average rating of the recipe          |
| `calories`       | Calories in the recipe                |
| `total_fat`      | Total fat content in the recipe       |
| `protein`        | Protein content in the recipe         |

---

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

First, I merged the recipes and interactions datasets using a left join to ensure that all recipes were included. I replaced 0 ratings with `NaN` to represent missing values. Next, I calculated the average rating for each recipe and added it as a new column (`rating_avg`) to serve as the target variable for prediction. I converted string columns like `tags`, `nutrition`, and `ingredients` into lists for feature extraction. Specifically, I splitted the `nutrition` column into separate attributes (e.g., `calories`, `total_fat`, `protein`) and converted them into floats for future investigation. The submitted column, initially stored as a string, was converted into a datetime object. Additionally, I categorized the `minutes` and `calories` columns into five bins and created new columns, `minutes_category` and `calories_category` respectively for future one-hot encoding.

For simplicity and readability, here is the head of my cleaned `recipes` DataFrame:

| **id**   | **minutes** | **tags**                                          | **n_steps** | **n_ingredients** | **rating_avg** | **calories** | **total_fat** | **minutes_category** | **calories_category** |
|----------|-------------|--------------------------------------------------|-------------|--------------------|----------------|--------------|---------------|----------------------|-----------------------|
| 333281   | 40          | [60-minutes-or-less ...]                         | 10          | 9                  | 4.0            | 138.4        | 10.0          | Medium               | Low                   |
| 453467   | 45          | [60-minutes-or-less ...]                         | 12          | 11                 | 5.0            | 595.1        | 46.0          | Medium               | Very High             |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |

---

### Univariate Analysis
This following bar chart with right-skewed reveals the most commonly used tags associated with recipes on Food.com. The most frequent tags often relate to preparation time, main ingredients, dietary attributes (such as "low-carb" or "low-sugar"), and required equipment. Understanding these tag frequencies helps in analyzing which types of recipes may achieve higher average ratings. For instance, people generally prefer healthy, tasty recipes that require minimal equipment and less preparation time, suggesting that recipes with such tags are likely to have higher average ratings. 

<iframe
  src="assets/frequency.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This following histogram with a left-skewed curve reveals that the majority of recipes(over 140k) have an average rating of 5. There is a significant drop in frequency for recipes with average ratings below 4, suggesting that highly-rated recipes dominate the dataset. This trend implies that users are more likely to favor popular, well-reviewed recipes and may be inclined to rate these recipes highly as well, reinforcing the dominance of high ratings in the dataset.

<iframe
  src="assets/distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### Bivariate Analysis
This following scatter plot shows the relationship between calories and total fat for recipes with calories under 500. There is a clear positive trend, indicating that recipes with higher calorie content within this range also tend to have higher amounts of total fat. This suggests that within lower-calories recipes, fat contributes significantly to the calorie count.

<iframe
  src="assets/scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This following bar chart shows the average ratings for recipes with the most common. Recipes with the most common tags, such as "15-minutes-or-less," "dietary," and "low-in-something," generally have high average ratings around 4.5 or above, indicating that popular recipe types focused on convenience, health, and common meal categories are well-received. This suggests that widely appealing recipe tags are associated with higher average ratings.

<iframe
  src="assets/tag2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates
In this pivot table, as the time required to prepare the recipe increases from "Very Quick" to "Very Slow," there is a noticeable trend of average ratings decreasing. Additionally, as the calorie category moves from "Very Low" to "Very High," there is also a general decline in average ratings, with an exception at the "Medium" preparation time, where ratings do not decrease as distinctly. This pattern suggests that recipes with shorter preparation times and lower calorie counts tend to receive higher ratings, possibly due to user preference for quicker and healthier recipes. In addition, recipes with a "Very Quick" preparation time and 'Low' calories tend to have the highest average ratings (4.75). This suggests that people prefer recipes with calorie counts between 100 and 200 that can be prepared quickly. 

| calories_category | Very Low | Low   | Medium | High  | Very High |
|--------------------|----------|-------|--------|-------|-----------|
| minutes_category   |          |       |        |       |           |
| Very Quick         | 4.74     | 4.75  | 4.72   | 4.68  | 4.66      |
| Quick              | 4.68     | 4.67  | 4.71   | 4.68  | 4.68      |
| Medium             | 4.61     | 4.65  | 4.68   | 4.68  | 4.67      |
| Slow               | 4.68     | 4.68  | 4.67   | 4.69  | 4.67      |
| Very Slow          | 4.68     | 4.66  | 4.64   | 4.61  | 4.63      |

---

### Imputation
The only column with missing values is `rating_avg`, intentionally left as `NaN`. After exploring Food.com, I observed that users can submit reviews without providing a rating, resulting in a rating value of 0. Since the actual rating scale on the site ranges from 1 to 5, a rating of 0 indicates that no rating was given, rather than a meaningful score. Therefore, we don’t need to fill in these missing values because they represent intentional omissions rather than incomplete data, accurately reflecting cases where no rating was assigned.

---

## Framing a Prediction Problem
My prediction problem focus on predicting the ratings of recipes. As my response variable, the column `ratings_avg` will contain continuous numbers, which implies that my prediction problem is a regression problem. I chose to predict recipe ratings because they are highly relevant to everyday life such that individuals often rely on ratings to decide which recipes to employ. Therefore, constructing a prediction model for recipe ratings is both practical and meaningful.

To evaluate my model, I employ RMSE as the primary metrics. I choose RMSE over R-Squared since R-squared tends to improve as model complexity increases, which will provide poor judgements on generalization over unseen data. Moreover, RMSE is measured in the similar unites as the response variable. Thus, I prefer RMSE over MSE due to its interpretability.

To train my model, I would consider columns including `minutes_category`, `n_steps`, `n_ingredients`, `calories_category`, and `total_fat` as my features. Those features are information that I would know at the “time of prediction”, which is essential in constructing my prediction model.

---

## Baseline Model
For my baseline model, I trained a **Linear Regression** model using both quantitative feature (`n_steps`) and ordinal feature (`calories_category`). In terms of feature engineering, I employed one-hot encoding to the ordinal column `calories_category`. After fitting the model and making predictions on unseen test data, the model achieved a test RMSE of around **0.49103**. Considering the target column, `rating_avg`, ranges from 1 to 5, this RMSE value is quite reasonable. Thus, I believe that my current baseline model is good enough since the prediction deviation is only around **0.4910**.

---

## Final Model
### New Features
`minutes_category`: Based on the plot analyzing the relationship between tags and average ratings, I observed that tags indicating preparation times (e.g. “30-minutes-or-less”) tend to have higher average ratings. To incorporate this intuition, I preprocessed the `minutes` column by dividing it into five categories as mentioned and created a new categorical feature named `minutes_category`.  

`n_steps` & `n_ingredients`: The analysis of tags also revealed that recipes with fewer preparation steps and ingredients will receive higher ratings. Intuitively, individuals will prefer convenient recipes with less number of steps and less complicated ingredients. Thus, I employed `n_steps` and `n_ingredients` as quantitative features. 

`calories_category`: People on diet will prefer recipes with tags such as "low-carb" or "low-in-something", as those tags also closely relate with recipe ratings. To capture this trend, I preprocessed the `calories` column into five bins and created a new categorical feature `calories_category`. 

`total_fat`: Intuitively, people who prioritize health will prefer recipes with lower fat content. By including `total_fat` as a feature, the model can better predict ratings based on the nutrition profile of recipes.

### Feature Engineering
Generally, I employed three feature engineering approaches on the selected features correspondingly. For quantitative features, through **PolynomialFeatures** from scikit-learn, I expanded `n_steps` and `n_ingredients` columns by generating high-order terms to capture non-linear patterns. Then, by using **StandardScaler**, I standardized those polynomial features, ensuring consistency during model training. Moreover, I standardized the `total_fat` feature to ensure it has the same scale as the transformed features. For categorical features, by utilizing **OneHotEncoder**, I performed one-hot encoding on both `minutes_category` and `calories_category` to further enhance model performance.

### Algorithm & Hyperparameter
I intended to enhance the model performance and prevent overfitting by employing **Ridge Regression** as my model algorithm. To better optimize, I tuned the polynomial degree for `n_steps` and `n_ingredients` features within the range of 1 to 4 to capture nonlinear relationships. Moreover, I tuned the regularization term (alpha) for Ridge Regression with values [0, 0.25, 0.5, 1, 2, 4]. Utilizing **GridSearchCV** with 5-fold cross-validation, the best hyperparameters were identified as a polynomial degree of 4 for both `n_steps` and `n_ingredients`, and alpha = 1 for L2 regularization term. 

### Model Performance
In conclusion, the overall model consists of polynomial transformation and standardization for quantitative features, one-hot encoding for categorical features, and Ridge Regression with optimal regularization term. With this architecture, I achieved a test RMSE of **0.4906**. In comparison, the baseline model, which employed only one-hot encoding on the calories_category feature and used Linear Regression as the algorithm, achieved a test RMSE of **0.4910**. My final model demonstrates a clear improvement over my baseline model, as the test RMSE decreased by **0.0004**.
