# Databricks notebook source
#defining the schema
from pyspark.sql.types import *

ratings_df_schema = StructType(
  [StructField('userId', IntegerType()),
   StructField('movieId', IntegerType()),
   StructField('rating', DoubleType())]
)
movies_df_schema = StructType(
  [StructField('ID', IntegerType()),
   StructField('title', StringType())]
)

# COMMAND ----------


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Movie_Recommendations").getOrCreate()
from pyspark.sql.functions import regexp_extract
from pyspark.sql.types import *

raw_ratings_df =  spark.read.csv('/FileStore/tables/ratings.csv',header = True ,  schema = ratings_df_schema)
ratings_df = raw_ratings_df.drop('Timestamp')

raw_movies_df = spark.read.csv('/FileStore/tables/movies.csv',header = True ,  schema = movies_df_schema)
movies_df = raw_movies_df.drop('Genres').withColumnRenamed('movieId', 'ID')

ratings_df.cache()
movies_df.cache()

assert ratings_df.is_cached
assert movies_df.is_cached

raw_ratings_count = raw_ratings_df.count()
ratings_count = ratings_df.count()
raw_movies_count = raw_movies_df.count()
movies_count = movies_df.count()

print ('There are %s ratings and %s movies in the datasets' % (ratings_count, movies_count))
print ('Ratings:')
ratings_df.show(3)
print ('Movies:')
movies_df.show(3, truncate=False)

assert raw_ratings_count == ratings_count
assert raw_movies_count == movies_count

# COMMAND ----------

# MAGIC %md Next, let's do a quick verification of the data.

# COMMAND ----------

assert ratings_count == 20000263
assert movies_count == 27278
assert movies_df.filter(movies_df.title == 'Toy Story (1995)').count() == 1
assert ratings_df.filter((ratings_df.userId == 6) & (ratings_df.movieId == 1) & (ratings_df.rating == 5.0)).count() == 1

# COMMAND ----------

display(movies_df)

# COMMAND ----------

display(ratings_df)

# COMMAND ----------

# MAGIC %md
# MAGIC One way to recommend movies is to always recommend the movies with the highest average rating. In this part, we will use Spark to find the name, number of ratings, and the average rating of the 20 movies with the highest average rating and at least 500 reviews. We want to filter our movies with high ratings but greater than or equal to 500 reviews because movies with few reviews may not have broad appeal to everyone.

# COMMAND ----------

from pyspark.sql import functions as F

# From ratingsDF, create a movie_ids_with_avg_ratings_df that combines the two DataFrames
movie_ids_with_avg_ratings_df = ratings_df.groupBy('movieId').agg(F.count(ratings_df.rating).alias("count"), F.avg(ratings_df.rating).alias("average"))
print ('movie_ids_with_avg_ratings_df:')
movie_ids_with_avg_ratings_df.show(3, truncate=False)


movie_names_df = movie_ids_with_avg_ratings_df.join(movies_df,movie_ids_with_avg_ratings_df.movieId == movies_df.ID)
movie_names_with_avg_ratings_df = movie_names_df.select(['average','title','count','movieId'])

print ('movie_names_with_avg_ratings_df:')
movie_names_with_avg_ratings_df.show(3, truncate=False)

# COMMAND ----------

# MAGIC %md Now that we have a DataFrame of the movies with highest average ratings, we can use Spark to determine the 20 movies with highest average ratings and at least 500 reviews.

# COMMAND ----------

movies_with_500_ratings_or_more = movie_names_with_avg_ratings_df.filter(F.col('count')>500).sort('count', ascending=False)
print ('Movies with highest ratings:')
movies_with_500_ratings_or_more.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md This is called as popularity based Recommendation.
# MAGIC Using a threshold on the number of reviews is one way to improve the recommendations, but there are many other good ways to improve quality. For example, you could weight ratings by the number of ratings.

# COMMAND ----------

# MAGIC %md We are going to use a technique called collaborative filtering. Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a person chosen randomly.

# COMMAND ----------

displayHTML("<img src ='data:image/png;base64,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'>")

# COMMAND ----------

# MAGIC %md
# MAGIC For movie recommendations, we start with a matrix whose entries are movie ratings by users (shown in red in the diagram below). Each column represents a user (shown in green) and each row represents a particular movie (shown in blue).
# MAGIC 
# MAGIC Since not all users have rated all movies, we do not know all of the entries in this matrix, which is precisely why we need collaborative filtering. For each user, we have ratings for only a subset of the movies. With collaborative filtering, the idea is to approximate the ratings matrix by factorizing it as the product of two matrices: one that describes properties of each user (shown in green), and one that describes properties of each movie (shown in blue).

# COMMAND ----------

# MAGIC %md
# MAGIC We want to select these two matrices such that the error for the users/movie pairs where we know the correct ratings is minimized. The Alternating Least Squares algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the movies such that the error is minimized. Then, it holds the movies matrix constant and optimizes the value of the user's matrix. This alternation between which matrix to optimize is the reason for the "alternating" in the name.
# MAGIC 
# MAGIC This optimization is what's being shown on the right in the image above. Given a fixed set of user factors (i.e., values in the users matrix), we use the known ratings to find the best values for the movie factors using the optimization written at the bottom of the figure. Then we "alternate" and pick the best user factors given fixed movie factors.

# COMMAND ----------

# MAGIC %md
# MAGIC Before we jump into using machine learning, we need to break up the ratings_df dataset into three pieces:
# MAGIC 
# MAGIC - A training set (DataFrame), which we will use to train models
# MAGIC - A validation set (DataFrame), which we will use to choose the best model
# MAGIC - A test set (DataFrame), which we will use for our experiments
# MAGIC - To randomly split the dataset into the multiple groups, we can use the pySpark randomSplit() transformation. randomSplit() takes a set of splits and a seed and returns multiple DataFrames.

# COMMAND ----------

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing

(split_60_df, split_a_20_df, split_b_20_df) = ratings_df.randomSplit([0.6,0.2,0.2])

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC In this part, we will use the Apache Spark ML Pipeline implementation of Alternating Least Squares, ALS. ALS takes a training dataset (DataFrame) and several parameters that control the model creation process. To determine the best values for the parameters, we will use ALS to train several models, and then we will select the best model and use the parameters from that model in the rest of the predictions

# COMMAND ----------

# MAGIC %md
# MAGIC Why are we doing our own cross-validation?
# MAGIC 
# MAGIC A challenge for collaborative filtering is how to provide ratings to a new user (a user who has not provided any ratings at all). Some recommendation systems choose to provide new users with a set of default ratings (e.g., an average value across all ratings), while others choose to provide no ratings for new users. Spark's ALS algorithm yields a NaN (Not a Number) value when asked to provide a rating for a new user.
# MAGIC 
# MAGIC Using the ML Pipeline's CrossValidator with ALS is thus problematic, because cross validation involves dividing the training data into a set of folds (e.g., three sets) and then using those folds for testing and evaluating the parameters during the parameter grid search process. It is likely that some of the folds will contain users that are not in the other folds, and, as a result, ALS produces NaN values for those new users. When the CrossValidator uses the Evaluator (RMSE) to compute an error metric, the RMSE algorithm will return NaN. This will make all of the parameters in the parameter grid appear to be equally good (or bad).
# MAGIC 
# MAGIC You can read the discussion on Spark JIRA 14489 about this issue. There are proposed workarounds of having ALS provide default values or having RMSE drop NaN values. Both introduce potential issues. We have chosen to have RMSE drop NaN values. While this does not solve the underlying issue of ALS not predicting a value for a new user, it does provide some evaluation value. We manually implement the parameter grid search process using a for loop (below) and remove the NaN values before using RMSE.
# MAGIC 
# MAGIC For a production application, you would want to consider the tradeoffs in how to handle new users.

# COMMAND ----------

from pyspark.ml.recommendation import ALS

# Let's initialize our ALS learner
als = ALS()

# Now we set the parameters for the method
als.setMaxIter(5)\
   .setRegParam(0.1)\
   .setUserCol("userId")\
  .setItemCol("movieId")\
  .setRatingCol("rating")

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12]
errors = [0, 0, 0]
models = [0, 0, 0]
err = 0
min_error = float('inf')
best_rank = -1
for rank in ranks:
  # Set the rank here:
  als.rank
  # Create the model with these parameters.
  model = als.fit(training_df)
  # Run the model to create a prediction. Predict against the validation_df.
  predict_df = model.transform(validation_df)

  # Remove NaN values from prediction (due to SPARK-14489)
  predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

  # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
  error = reg_eval.evaluate(predicted_ratings_df)
  errors[err] = error
  models[err] = model
  print ('For rank %s the RMSE is %s' % (rank, error))
  if error < min_error:
    min_error = error
    best_rank = err
  err += 1

als.setRank(ranks[best_rank])
print ('The best model was trained with rank %s' % ranks[best_rank])
my_model = models[best_rank]

# COMMAND ----------

# MAGIC %md
# MAGIC So far, we used the training_df and validation_df datasets to select the best model. Since we used these two datasets to determine what model is best, we cannot use them to test how good the model is; otherwise, we would be very vulnerable to overfitting. To decide how good our model is, we need to use the test_df dataset. We will use the best_rank you determined in part (2b) to create a model for predicting the ratings for the test dataset and then we will compute the RMSE.

# COMMAND ----------

predict_df = my_model.transform(test_df)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df )

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

# COMMAND ----------

predicted_test_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the RMSE for the results predicted by the model versus the values in the test set is one way to evalute the quality of our model. Another way to evaluate the model is to evaluate the error from a test set where every rating is the average rating for the training set.

# COMMAND ----------

avg_rating_df = training_df.agg({"rating": "avg"})

# Extract the average rating value. (This is row 0, column 0.)
training_avg_rating = avg_rating_df.collect()[0][0]

print('The average rating for movies in the training set is {0}'.format(training_avg_rating))

# Add a column with the average rating
test_for_avg_df = test_df.withColumn('prediction', F.lit(training_avg_rating))

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = reg_eval.evaluate(test_for_avg_df)

print("The RMSE on the average set is {0}".format(test_avg_RMSE))


# COMMAND ----------

# MAGIC %md
# MAGIC Our model is performing better than the Baseline model and we have created a Recommender system for the users :)
