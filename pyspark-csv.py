from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StringType


spark = SparkSession.builder \
                    .appName("ctr prediction") \
                    .getOrCreate()

impression = spark.read.option('header', 'true') \
                       .option('inferSchema', 'true') \
                       .csv('gs://pyspark-yli/avazu-ctr-prediction/train.csv') \
                       .selectExpr("*", "substr(hour, 7) as hr") \
                       .drop('_c0')

impression = impression.withColumn('hr-app_category', F.concat('hr', 'app_category')) \
                       .withColumn('hr-site_category', F.concat('hr', 'site_category')) \
                       .withColumn('hr-device_type', F.concat('hr', 'device_type')) \
                       .withColumn('banner_pos-device_type', F.concat('banner_pos', 'device_type')) \
                       .withColumn('device_type-app_category', F.concat('device_type', 'app_category')) \
                       .withColumn('device_type-site_category', F.concat('device_type', 'site_category'))

strCols = map(lambda t: t[0], filter(lambda t: t[1] == 'string', impression.dtypes))
intCols = map(lambda t: t[0], filter(lambda t: t[1] == 'int', impression.dtypes))

# [row_idx][json_idx]
strColsCount = sorted(map(lambda c: (c, impression.select(F.countDistinct(c)).collect()[0][0]), strCols), key=lambda x: x[1], reverse=True)
intColsCount = sorted(map(lambda c: (c, impression.select(F.countDistinct(c)).collect()[0][0]), intCols), key=lambda x: x[1], reverse=True)

# All of the columns (string or integer) are categorical columns
#  except for the [click] column
maxBins = 100
wide_cols = list(map(lambda c: c[0], filter(lambda c: c[1] <= maxBins, strColsCount)))
wide_cols += list(map(lambda c: c[0], filter(lambda c: c[1] <= maxBins, intColsCount)))
wide_cols.remove('click')

embed_cols = [('device_model', impression.select('device_model').distinct().count(), 256),
              ('app_id', impression.select('app_id').distinct().count(), 256),
              ('site_id', impression.select('site_id').distinct().count(), 256),
              ('site_domain', impression.select('site_domain').distinct().count(), 256),
              ('app_domain', impression.select('app_domain').distinct().count(), 128),
             ]

# Apply string indexer to all of the categorical columns
#  And add _idx to the column name to indicate the index of the categorical value
strIndexers_wide = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+'_idx'), wide_cols))

embed_features = map(lambda c:c[0] + 'SEP' + str(c[1]) + 'SEP' + str(c[2]), embed_cols)
strIndexers_embed = list(map(lambda c: StringIndexer(inputCol=c[0],
                                                     outputCol=c[0] + 'SEP' + str(c[1]) + 'SEP' + str(c[2])
                                                    ), embed_cols))
oneHotEncoders = list(map(lambda c: OneHotEncoder(inputCol=c+'_idx', outputCol=c+'_onehot'), wide_cols))

# Assemble the put as the input to the VectorAssembler 
#   with the output being our features
vectorAssembler = VectorAssembler(inputCols=list(map(lambda c: c+'_onehot', wide_cols)), outputCol='wide_features_v')
# The [click] column is our label 
labelStringIndexer = StringIndexer(inputCol = "click", outputCol = "label")
# The stages of our ML pipeline 
stages = strIndexers_wide + strIndexers_embed + oneHotEncoders + [vectorAssembler, labelStringIndexer]

# Create our pipeline
pipeline = Pipeline(stages = stages)

# create transformer to add features
featurizer = pipeline.fit(impression)

def string_from_array(input_list):
    return ('[' + ','.join([str(item) for item in input_list]) + ']')

ats_udf = F.udf(string_from_array, StringType())


# dataframe with feature and intermediate transformation columns appended
featurizedImpressions = featurizer.transform(impression) \
                                  .withColumn('wide_features', ats_udf(vector_to_array('wide_features_v')))                                

train, test = featurizedImpressions.select('wide_features', 'label', *embed_features) \
                                   .randomSplit([0.7, 0.3], 42)

# train_repartition = train.count() // 100000
# val_repartition = val.count() // 100000

train.repartition(1) \
     .write \
     .mode('overwrite') \
     .csv('gs://pyspark-yli/avazu-ctr-prediction/training_csv')

test.repartition(1) \
    .write \
    .mode('overwrite') \
    .csv('gs://pyspark-yli/avazu-ctr-prediction/validation_csv')

spark.stop()

