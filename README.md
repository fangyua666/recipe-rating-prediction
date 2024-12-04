# Recipe Ratings Prediction

This project aims to predict recipe ratings using data from Food.com. The learning pipeline follows a standard machine learning framework, beginning with comprehensive data cleaning and exploratory data analysis. The project frames a prediction problem focused on predicting recipe ratings and systematically enhances model performance through development of optimized model. 

The project, authored by Fang Yu, was conducted at University of Michigan.

---

## Introduction

The dataset, sourced from Food.com, contains recipes and ratings posted since 2008. It includes detailed information about recipes, such as preparation times, nutrition information, and user ratings. The focus of this project is to predict recipe ratings and explore the question: what types of recipes tend to receive higher ratings? I intend to utilize data analysis techniques (e.g. pandas and numpy) to evaluate the relationship between ratings and various columns. Based on the information derived from exploratory data analysis, I can construct a prediction model to predict the ratings of recipes. Specifically, recipe rating is a crucial factor for users to decide which recipe to follow. These insights can help users select recipes that highly align with their preferences. With a total of 234,429 rows, this dataset becomes a perfect choice for constructing such a prediction model. Here is a brief descriptions to each of the relevant columns:

| **Column**         | **Description**                        |
|---------------------|----------------------------------------|
| `'name'`           | Recipe name                           |
| `'id'`             | Recipe ID                             |
| `'minutes'`        | Minutes to prepare the recipe         |
| `'submitted'`      | Date the recipe was submitted         |
| `'tags'`           | Food.com tags for the recipe          |
| `'n_steps'`        | Number of steps in the recipe         |
| `'ingredients'`    | List of ingredients used in the recipe|
| `'n_ingredients'`  | Number of ingredients in the recipe   |
| `'rating_avg'`     | Average rating of the recipe          |
| `'calories'`       | Calories in the recipe                |
| `'total_fat'`      | Total fat content in the recipe       |
| `'protein'`        | Protein content in the recipe         |

---

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

First, I merged the recipes and interactions datasets using a left join to ensure that all recipes were included. I replaced `'0'` ratings with `'NaN'` to represent missing values. Next, I calculated the average rating for each recipe and added it as a new column (`'rating_avg'`) to serve as the target variable for prediction. I converted string columns like `'tags'`, `'nutrition'`, and `'ingredients'` into lists for feature extraction. Specifically, I splitted the `'nutrition'` column into separate attributes (e.g., `'calories'`, `'total_fat'`, `'protein'`) and converted them into floats for future investigation. The submitted column, initially stored as a string, was converted into a datetime object. Additionally, I categorized the `'minutes'` and `'calories'` columns into five bins and created new columns, `'minutes_category'` and `'calories_category'` respectively for future one-hot encoding.

For simplicity and readability, here is the head of my cleaned `'recipes'` DataFrame:

| **id**   | **minutes** | **tags**                                          | **n_steps** | **n_ingredients** | **rating_avg** | **calories** | **total_fat** | **minutes_category** | **calories_category** |
|----------|-------------|--------------------------------------------------|-------------|--------------------|----------------|--------------|---------------|----------------------|-----------------------|
| 333281   | 40          | [60-minutes-or-less ...]                         | 10          | 9                  | 4.0            | 138.4        | 10.0          | Medium               | Low                   |
| 453467   | 45          | [60-minutes-or-less ...]                         | 12          | 11                 | 5.0            | 595.1        | 46.0          | Medium               | Very High             |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |
| 306168   | 40          | [60-minutes-or-less ...]                         | 6           | 9                  | 5.0            | 194.8        | 20.0          | Medium               | Low                   |

---

### Univariate Analysis
This following histogram with a left-skewed curve reveals that the majority of recipes(over 140k) have an average rating of 5. There is a significant drop in frequency for recipes with average ratings below 4, suggesting that highly-rated recipes dominate the dataset. This trend implies that users are more likely to favor popular, well-reviewed recipes and may be inclined to rate these recipes highly as well, reinforcing the dominance of high ratings in the dataset.

<iframe
  src="assets/frequency.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This following bar chart with right-skewed reveals the most commonly used tags associated with recipes on Food.com. The most frequent tags often relate to preparation time, main ingredients, dietary attributes (such as "low-carb" or "low-sugar"), and required equipment. Understanding these tag frequencies helps in analyzing which types of recipes may achieve higher average ratings. For instance, people generally prefer healthy, tasty recipes that require minimal equipment and less preparation time, suggesting that recipes with such tags are likely to have higher average ratings. 

<iframe
  src="assets/distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This following scatter plot shows the relationship between calories and total fat for recipes with calories under 500. There is a clear positive trend, indicating that recipes with higher calorie content within this range also tend to have higher amounts of total fat. This suggests that within lower-calorie recipes, fat contributes significantly to the calorie count.

<iframe
  src="assets/scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This following bar chart shows the average ratings for recipes with the most common, intermediate, and least common tags. Recipes with the most common tags, such as "15-minutes-or-less," "dietary," and "low-in-something," generally have high average ratings around 4.5 or above, indicating that popular recipe types focused on convenience, health, and common meal categories are well-received. Recipes with intermediate tags show a similar trend, maintaining relatively high ratings. However, recipes with the least common tags display more variability, with some tags achieving ratings near 5 and others much lower, suggesting that niche or specific recipe characteristics evoke mixed responses from users. This indicates that widely appealing recipe tags are associated with higher average ratings, while less common tags are more polarizing. To answer the question, the features will be subset from the most common tags, since it provide generally a stable high rating.

<iframe
  src="assets/tag.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


