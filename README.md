# Nest ML: No Place Like Home

## Introduction

**No Place Like Home** is a full-stack web application that helps discover areas across the United States that share key traits with your hometown. By entering your location, the model utilizes U.S. government data to find places with similar demographics and urban characteristics, assisting in exploring new destinations that feel like home.

[**Live Site**](https://nest.ahmedawad.io/)

## Features

- **Personalized Recommendations**: Find locations across the country that match your hometown's key characteristics.
- **Data-Driven Insights**: Leverage U.S. Census data to compare demographics and urban features.
- **Interactive Interface**: User-friendly platform built with modern web technologies.
- **Visualizations**: Dynamic charts and graphs using Shadcn charts.
- **AI-Generated Images**: Pastel images of metropolitan areas generated using DALL·E 3.

## Technology Stack

- **Frontend**: Next.js, React, TypeScript
- **Backend**:
  - **Database**: Supabase (PostgreSQL)
  - **Storage**: Supabase Bucket Storage for images
- **Edge Functions**: Next.js Edge Functions using Flask in Python
- **Machine Learning Model**: KMeans clustering algorithm
- **Data Science and AI/ML**: Extensive data analysis and modeling
- **Visualizations**: Shadcn charts for interactive data presentation

## Data Sources

- **U.S. Census American Community Survey (ACS)**: Provides detailed demographic, housing, and economic data.
- **DataUSA.io**: Offers descriptions and additional context for various locations.

## Machine Learning Model Overview

The model identifies similarities between metropolitan areas using clustering techniques on various demographic and urban features.

### Key Steps:

1. **Data Loading and Preparation**:
   - Load data from the U.S. Census ACS survey.
   - Clean the dataset by handling missing values and filtering out features with a high percentage of missing data.

2. **Data Scaling**:
   - Standardize features using `StandardScaler` to ensure each has a mean of zero and unit variance.

3. **Dimensionality Reduction**:
   - Apply Principal Component Analysis (PCA) to reduce dimensionality while retaining 95% of the variance.
   - Enhances computational efficiency and model performance.

4. **Clustering**:
   - Use the KMeans algorithm to cluster locations based on the processed features.
   - Determine the optimal number of clusters using the Elbow Method and Silhouette Analysis.

5. **Nearest Neighbors Identification**:
   - For each location, find the nearest neighbors within its cluster.
   - If insufficient neighbors are found, extend the search to neighboring clusters based on centroid proximity.

6. **Feature Importance Analysis**:
   - Employ a Random Forest Classifier to identify the most significant features contributing to the clustering.
   - Top features include housing age, median rent, poverty levels, and more.

## Data Processing and Analysis

Data processing and model training involve extensive use of data science and AI/ML techniques, utilizing the following key libraries:

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms for scaling, PCA, clustering, and classification
- **Matplotlib** and **Seaborn**: Data visualization
- **Requests** and **BeautifulSoup**: Web scraping
- **Asyncio** and **Playwright**: Asynchronous web scraping

## Data Sources and Licensing

- **U.S. Census ACS Survey Data**:
  - Provides comprehensive demographic, housing, and economic information.
  - Data is accessed via the [U.S. Census API](https://www.census.gov/data/developers/data-sets/acs-5year.html).

- **DataUSA.io**:
  - Offers detailed narratives and statistics for various locations.
  - Data usage complies with [DataUSA's Terms of Service](https://datausa.io/about/usage).

## Acknowledgements

- **U.S. Census Bureau**: For providing open access to valuable data.
- **DataUSA.io**: For offering comprehensive location descriptions.
- **Acceternity**: For the beautiful animations.
- **Open-Source Community**: For the tools and libraries that made this project possible.

## More to Come Soon

Continuously working on improving the model and adding new features. Stay tuned for updates!
