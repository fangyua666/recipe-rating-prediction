# Recipe Ratings Prediction

This project aims to predict recipe ratings using data from Food.com. The learning pipeline follows a standard machine learning framework, beginning with comprehensive data cleaning and exploratory data analysis. The project frames a prediction problem focused on predicting recipe ratings and systematically enhances model performance through development of optimized model. 

The project, authored by Fang Yu, was conducted at University of Michigan.

---

## Introduction

The dataset, sourced from Food.com, contains recipes and ratings posted since 2008. It includes detailed information about recipes, such as preparation times, nutrition information, and user ratings. The focus of this project is to predict recipe ratings and explore the question: what types of recipes tend to receive higher ratings? I intend to utilize data analysis techniques (e.g. pandas and numpy) to evaluate the relationship between ratings and various columns. Based on the information derived from exploratory data analysis, I can construct a prediction model to predict the ratings of recipes. Specifically, recipe rating is a crucial factor for users to decide which recipe to follow. These insights can help users select recipes that highly align with their preferences. With a total of 234,429 rows, this dataset becomes a perfect choice for constructing such a prediction model. Here is a brief descriptions to each of the relevant columns:

|Column                |Description|
|---                |---        |
|`'name'`                |Recipe name|
|`'id'`                |Recipe ID|
|`'minutes'`                |Minutes to prepare recipe|
|`'submitted'`                |Date recipe was submitted|
|`'tags'`                |Food.com tags for recipe|
|`'n_steps'`                |Number of steps in recipe|
|`'ingredients'`                |Ingredients of recipe|
|`'n_ingredients'`                |Number of ingredients in recipe|
|`'rating_avg'`                |Average rating of recipe|
|`'calories'`                |Calories of recipe|
|`'total_fat'`                |Total fat amount of recipe|
|`'protein'`                |Protein of recipe|
---

## Data Cleaning and Exploratory Data Analysis