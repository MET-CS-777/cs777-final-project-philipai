from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score#, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


class Metropolis(Enum):
    Atlanta = "Atlanta, GA"
    Baltimore = "Baltimore, MD"
    Boston = "Boston, MA"
    Chicago = "Chicago, IL"
    Dallas = "Dallas, TX"
    Denver = "Denver, CO"
    Detroit = "Detroit, MI"
    Houston = "Houston, TX"
    Los_Angeles = "Los Angeles, CA"
    Miami = "Miami, FL"
    Minneapolis = "Minneapolis, MN"
    New_York = "New York, NY"
    Philadelphia = "Philadelphia, PA"
    Phoenix = "Phoenix, AZ"
    San_Diego = "San Diego, CA"
    San_Francisco = "San Francisco, CA"
    Seattle = "Seattle, WA"
    St_Louis = "St. Louis, MO"
    Tampa = "Tampa, FL"
    Washington = "Washington, DC"

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: main.py <Home_Sales_Price.csv> <Household_Income.csv> <US_Cities.csv> <Major_Cities_CPI.csv>", file=sys.stderr)
        exit(-1)

    ############ Initialization: Cloud Environment
    spark = SparkSession.builder.getOrCreate()
    sc = SparkContext.getOrCreate()

    print(f"Spark Version: {spark.sparkContext.version}")
    print(f"Python Version: {spark.sparkContext.pythonVer}")
    print(f"Spark Context Master: {spark.sparkContext.master}")




    ############ Load Datasets
    data_home_sales = sys.argv[1]
    data_household_income = sys.argv[2]
    data_census = sys.argv[3]
    data_cpi = sys.argv[4]

    df_home_sales = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(data_home_sales)
    df_household_income_expenditures = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(data_household_income)
    df_census = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(data_census)
    df_cpi = spark.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(data_cpi) 


    df_home_sales.show()
    df_household_income_expenditures.show()
    df_census.show()
    df_cpi.show()




    ############ Assemble all data sources and then Render a Consolidated dataframe
    metropolis_column = "Metropolis"
    median_real_estate_sales_price_column = "Median_Real_Estate_Sales_Price"
    median_household_income_column = "Median_Household_Income"
    cost_of_living_column = "Cost_of_Living"
    cpi_column = "Consumer_Price_Index"
    population_density_column = "Population_Density"

    columns = StructType([  StructField(metropolis_column, StringType(), True),
                            StructField(median_real_estate_sales_price_column, FloatType(), True),
                            StructField(median_household_income_column, IntegerType(), True),
                            StructField(cost_of_living_column, IntegerType(), True),
                            StructField(cpi_column, FloatType(), True),
                            StructField(population_density_column, FloatType(), True) ] )

    # Build consolidated dataframe for further analysis
    df_consolidated = spark.createDataFrame(data=[], schema=columns)


    # Choose relevant data per city(metropolis)
    for metropolis in list(Metropolis):
        print(f"Begin to Process {metropolis.value} data......")
        city_and_state = metropolis.value
        city = metropolis.value.split(", ")[0]
        state_id = metropolis.value.split(", ")[1]
        number_of_data_points = 32

        # Fetch the individual data batch from the record
        city_repetitions = [city_and_state]*number_of_data_points

        house_sales_per_city = list(df_home_sales.filter(df_home_sales.RegionName == city_and_state).first()[-number_of_data_points:])
        house_sales_per_city = list(filter(None, house_sales_per_city) )
        real_estate_confidence_interval = stats.t.interval(confidence=0.95, df=len(house_sales_per_city)-1, loc=np.mean(house_sales_per_city), scale=stats.sem(house_sales_per_city))


        income_per_city = [df_household_income_expenditures.filter((df_household_income_expenditures.Metropolis == city) & (df_household_income_expenditures.FIPS_Code == state_id) ).select("Median_Household_Income").first()[0]]*number_of_data_points
        
        expenditures_per_city = [df_household_income_expenditures.filter((df_household_income_expenditures.Metropolis == city) & (df_household_income_expenditures.FIPS_Code == state_id) ).select("Cost_of_Living").first()[0]]*number_of_data_points
                                    
        cpi_bimonthly = df_cpi.select(city_and_state.replace(".", "")).collect()
        cpi_per_city = np.repeat(cpi_bimonthly, 8) # Make up 32 elements totally
        cpi_per_city = [float(x) for x in cpi_per_city] # Pyspark does not support Numpy datatype


        city_population_densities = [ df_census.filter((df_census.city == city) & (df_census.state_id == state_id)) \
                                            .select("density").first()[0] ]*number_of_data_points
        
        
        # Aggregate values by rows
        added_rows_per_city = zip(city_repetitions, house_sales_per_city, income_per_city, expenditures_per_city,
                                cpi_per_city, city_population_densities)
        
        # Save data set by metropolis into a consolidated list
        added_df_per_city = spark.createDataFrame(added_rows_per_city, columns)
        df_consolidated = df_consolidated.union(added_df_per_city)
        print(f"Complete processing {metropolis.value} data.")


    df_consolidated = df_consolidated.dropna()
    df_consolidated.show()
    df_consolidated.cache() # save it in the fast-access memory



    ############ One-Way ANOVA(Analysis of variance)
    print("Compare Real Estate Median Sales Price across major metropolises")
    ordinary_least_squares_model = ols(f"{median_household_income_column} ~ {metropolis_column}", data=df_consolidated.toPandas()).fit()
    anova_table_real_estates = sm.stats.anova_lm(ordinary_least_squares_model, typ=1)
    print(anova_table_real_estates)
    if anova_table_real_estates['PR(>F)'][0] >- 0.05:
        print("There must be house price differences between one pair of citis at least, based on 95% confidence level")
    else:
        print("There is no significant house price difference between one pair of citis at least, based on 95% confidence level")


    print("Compare Consumer Price Index(CPI) across major metropolises")
    ordinary_least_squares_model2 = ols(f"{cpi_column} ~ {metropolis_column}", data=df_consolidated.toPandas()).fit()
    anova_table_cpi = sm.stats.anova_lm(ordinary_least_squares_model2, typ=1)
    print(anova_table_cpi)
    if anova_table_cpi['PR(>F)'][0] >- 0.05:
        print("There must be CPI differences between one pair of citis at least, based on 95% confidence level")
    else:
        print("There is no significant CPI difference between one pair of citis at least, based on 95% confidence level")




    ############ Tukey's Honestly Significant Difference(HSD) Multiple Comparisons Test
    # Compare Real Estate Prices in paired major metropolises
    tukey_real_estate = pairwise_tukeyhsd(endog=df_consolidated.toPandas()[[median_real_estate_sales_price_column]].values,
                            groups=df_consolidated.toPandas()[[metropolis_column]].values,
                            alpha=0.05)

    print(tukey_real_estate)

    # Compare Consumer Price Index(CPI) in paired major metropolises
    tukey_CPI = pairwise_tukeyhsd(endog=df_consolidated.toPandas()[[cpi_column]].values,
                            groups=df_consolidated.toPandas()[[metropolis_column]].values,
                            alpha=0.05)
    print(tukey_CPI)




    ############ Linear Regression With PySpark MLlib
    # Build the model
    assembler = VectorAssembler(inputCols=[median_real_estate_sales_price_column, median_household_income_column, cpi_column, population_density_column], outputCol="features")
    consolidated_data = assembler.transform(df_consolidated)
    train_data, test_data = consolidated_data.randomSplit([0.80, 0.20], seed=10)
    regression_model = LinearRegression(featuresCol='features', 
                                        labelCol=cost_of_living_column, maxIter=10, regParam=0.1).fit(train_data)

    # Print the model
    print(f"Coefficients: {regression_model.coefficients}")
    print(f"Intercept: {regression_model.intercept}")
    print(f"Linear Equation: CostOfLiving = ({regression_model.coefficients[0]}*{median_household_income_column}) + ({regression_model.coefficients[1]}*{cpi_column}) + ({regression_model.coefficients[2]}*{population_density_column}) + ({regression_model.intercept})")

    # Test the model
    predictions_cost_of_living = regression_model.transform(test_data)
    # print(predictions_cost_of_living.toPandas().head(5))

    # Evaluate the model
    evaluator1 = RegressionEvaluator(labelCol=cost_of_living_column, predictionCol='prediction', metricName='rmse')
    evaluator2 = RegressionEvaluator(labelCol=cost_of_living_column, predictionCol='prediction', metricName='r2')
    rmse = evaluator1.evaluate(predictions_cost_of_living)
    r2 = evaluator2.evaluate(predictions_cost_of_living)
    print("Root Mean Squared Error(RMSE):", rmse)
    print("R-squared(R2):", r2)



    ############ Multiple Linear Regression with SKlearn
    # Build the model
    Xs = df_consolidated.toPandas()[[median_real_estate_sales_price_column, median_household_income_column, cpi_column, population_density_column]].values
    Y = df_consolidated.toPandas()[cost_of_living_column].values
    Xs_train, Xs_test, Y_train, Y_test = train_test_split(Xs, Y, test_size=0.2, random_state=10)

    # Scale the features
    scaler = StandardScaler()
    Xs_train_scaled = scaler.fit_transform(Xs_train)
    Xs_test_scaled = scaler.transform(Xs_test)

    # Train the Model
    regression_model = linear_model.LinearRegression()
    regression_model.fit(Xs_train_scaled, Y_train)

    # Print the model
    print("Coefficients:", regression_model.coef_)
    print("Intercept:", regression_model.intercept_)
    print(f"Linear Equation: CostOfLiving = ({regression_model.coef_[0]}*{median_household_income_column}) + ({regression_model.coef_[1]}*{cpi_column}) + ({regression_model.coef_[2]}*{population_density_column}) + ({regression_model.intercept_})")

    # Test the regression model and Make predictions
    predictions_cost_of_living_scaled = regression_model.predict(Xs_test_scaled)

    # Evaluate the model
    print("Root Mean Squared Error(RMSE):", mean_squared_error(Y_test, predictions_cost_of_living_scaled, squared=False))
    print("R-squared(R2):", r2_score(Y_test, predictions_cost_of_living_scaled))
    Xs = sm.add_constant(scaler.transform(Xs))
    results = sm.OLS(Y, Xs).fit()
    results.summary()





    ############ Visualizations (Compare Purchase Power Nationwide)
    items_per_batch = 5
    batches = df_household_income_expenditures.count()/items_per_batch

    # Draw Paried Incomes and Cost of living bars in batches
    for index in range(0, int(batches), 1):
        label_x = "Metropolitan Regions"
        label_y = "US Dollars($)"
        upper_limit = 165000
        interval = 5000

        df_batch_data = df_household_income_expenditures.toPandas().iloc[index*items_per_batch:(index*items_per_batch)+items_per_batch]
        X_axis = np.arange(len(df_batch_data["Metropolis"]))

        delta_balances_batch = np.array(df_batch_data["Median_Household_Income"]) - np.array(df_batch_data["Cost_of_Living"])
        
        # Comparison Plot
        df_batch_data.plot(x='Metropolis', y=['Median_Household_Income', 'Cost_of_Living'], kind='bar', width = 0.4)
        plt.xticks(X_axis, df_batch_data["Metropolis"])
        plt.yticks(range(0, upper_limit, interval))
        plt.grid(axis='y')  
        plt.xlabel(label_x) 
        plt.ylabel(label_y) 
        plt.title("Comparison of Financial Health Among US Residents (2023)") 
        plt.legend() 
        plt.show()


        # Delta Plot
        plt.xticks(X_axis, df_batch_data["Metropolis"])
        plt.bar(X_axis, delta_balances_batch, width = 0.4, label = "Remaining Balance")
        plt.grid(axis='y') 
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title("Comparison of Purchase Power Among US Residents (2023)")
        plt.legend() 
        plt.show()





    ############ Visualizations (Display Summary Statistics of CPI and Real Estates Cost Per City)
    batches = len(list(Metropolis))/items_per_batch

    for index in range(0, int(batches), 1):
        label_x = "Metropolitan Regions"

        metropolises_batch = list(df_consolidated.toPandas().groupby([metropolis_column]).groups.keys())[index*items_per_batch:(index*items_per_batch)+items_per_batch]
        df_batch_data = df_consolidated.toPandas().groupby([metropolis_column]).filter(lambda x: x.name in metropolises_batch) 
        X_axis = np.arange(1, len(metropolises_batch)+1, 1)
        
        # Real Restate Prices
        df_batch_data.boxplot(column=median_real_estate_sales_price_column, by=metropolis_column, notch=False, patch_artist=True)
        plt.xticks(X_axis, metropolises_batch)
        plt.xlabel(label_x) 
        plt.ylabel("US Dollars($)") 
        plt.title("Comparison of Real Estates Prices Across US Metropolises") 
        plt.legend(["Real Estates Price"]) 
        plt.show()

        # CPI
        df_batch_data.boxplot(column=cpi_column, by=metropolis_column, notch=False, patch_artist=True)
        plt.xticks(X_axis, metropolises_batch)
        plt.xlabel(label_x) 
        plt.ylabel("Consumer Price Index (CPI)") 
        plt.title("Comparison of Consumer Price Index (CPI) Across US Metropolises") 
        plt.legend(["Consumer Price Index (CPI)"]) 
        plt.show()