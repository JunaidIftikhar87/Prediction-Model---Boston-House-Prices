import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


## Step -1: Specify the path to your CSV file
file_path = r'C:\Users\junai\Downloads\Northeastern\Quarters\Quarter 2\ALY 6015\Assignments\A1\Dataset\AmesHousing.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


## Step -2: Descriptive Statistics

# Explore data types and missing values
data_info = df.info()

# Select numerical columns for EDA (exclude non-numeric columns)
numerical_cols = df.select_dtypes(include=['number'])

# Generate summary statistics for numerical variables
summary_stats = numerical_cols.describe()

# Export summary statistics to a CSV file
#summary_stats.to_csv('summary_statistics.csv')

## Step -3: Data Cleaning

# Check for missing values
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Assuming 'df' is your DataFrame
df_numeric_columns = df.select_dtypes(include='number')

##Step-4
correlation_matrix = df_numeric_columns.corr()
print(correlation_matrix)

# Calculate the correlation matrix
correlation_matrix = df_numeric_columns.corr()

# Set the threshold for correlation
threshold = 0.5

##Step -5
# Create an empty list to store variable pairs with correlation > threshold
correlated_variables = []

# Iterate through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            # Add the variable pair to the list
            correlated_variables.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

# Print the correlated variable pairs
for var1, var2 in correlated_variables:
    print(f"{var1} and {var2} have a correlation greater than {threshold}")

# Define the filename for the export
Selected_Variables = "correlated_variables.csv"

# Open the file for writing
with open(Selected_Variables, "w") as output_file:
    # Write the correlated variable pairs to the file
    for var1, var2 in correlated_variables:
        output_file.write(f"{var1},{var2}\n")

print(f"Correlated variables have been exported to '{Selected_Variables}'.")



# Your code to identify correlated variables here...

# Create a DataFrame containing only the correlated variables
correlated_df = df[list(set(var1 for var1, var2 in correlated_variables))]

# Create a correlation matrix for the correlated variables
correlation_matrix = correlated_df.corr()

# Create a heatmap of the correlation matrix with adjusted axis labels
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Adjust the x and y axis labels
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.title("Correlation Heatmap of Correlated Variables")
plt.tight_layout()  # Ensures that the labels fit within the display area
plt.show()

#Step -6
# Highest Correl Variable
# Specify the columns you want to plot
x = df['Gr Liv Area']
y = df['SalePrice']

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)  # 'alpha' controls point transparency
plt.title('Scatter Plot of Gross Living Area vs. Sale Price')
plt.xlabel('Gross Living Area')
plt.ylabel('Sale Price')
plt.grid(True)  # Add grid lines if desired

# Show the plot
plt.show()

# Lowest Correl Variable
# Specify the columns you want to plot
x = df['Yr Sold']
y = df['SalePrice']

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)  # 'alpha' controls point transparency
plt.title('Scatter Plot of Year Sold vs. Sale Price')
plt.xlabel('Year Sold')
plt.ylabel('Sale Price')
plt.grid(True)  # Add grid lines if desired

# Show the plot
plt.show()

# Close to 0.5 Correl Variable
# Specify the columns you want to plot
x = df['TotRms AbvGrd']
y = df['SalePrice']

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)  # 'alpha' controls point transparency
plt.title('Scatter Plot of Total Rooms Above Ground vs. Sale Price')
plt.xlabel('Total Rooms Above Ground')
plt.ylabel('Sale Price')
plt.grid(True)  # Add grid lines if desired

# Show the plot
plt.show()

##Step- 7
X = df[['Gr Liv Area', 'Year Built', 'Mas Vnr Area']]
y = df['SalePrice']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Export the summary table to a CSV file
summary_df = pd.DataFrame({'Parameter': model.params, 'Std. Err.': model.bse, 't-value': model.tvalues, 'P-value': model.pvalues})
#summary_df.to_csv('model_summary.csv')

# Assuming you've already run the regression model and have the 'model' object

# Get the coefficients from the model summary
coefficients = model.params


# Creating a regression equation string
regression_equation = f"SalePrice = {coefficients['const']:.2f} + " \
                      f"{coefficients['Gr Liv Area']:.2f} * Gr Liv Area + " \
                      f"{coefficients['Mas Vnr Area']:.2f} * Mas Vnr Area + " \
                      f"{coefficients['Year Built']:.2f} * Year Built"

# Print the regression equation
print("Regression Equation:")
print(regression_equation)


# Get the predicted values from the model
predicted_values = model.predict(X)

# Create a scatterplot of actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=predicted_values)
plt.title('Actual vs. Predicted SalePrice')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')

# Add a diagonal line for reference (perfect prediction)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[['Gr Liv Area', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Garage Yr Blt', 'Garage Cars', 'Garage Area']]
X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Export VIF results to a CSV file
#vif.to_csv('vif_results.csv', index=False)
print(vif)


