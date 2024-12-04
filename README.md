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
<iframe
  src="assets/frequency.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
---

<iframe
  src="assets/distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>