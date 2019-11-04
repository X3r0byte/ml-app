using System;
using System.Collections.Generic;
using DataSet;
using Microsoft.ML.Data;
using Microsoft.ML;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;
using System.IO;

namespace Application
{
    public class BinaryPrediction: BinaryData
    {
        public static MLContext mlContext { get; set; }

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }

        private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model-binary.zip");
        public static ITransformer model;
        public static DataViewSchema schema;

        public TrainTestData LoadData(MLContext mlContext, string data)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<BinaryData>(data, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            schema = dataView.Schema;
            return splitDataView;
        }

        public ITransformer BuildAndTrainModel(MLContext context, IDataView splitTrainSet)
        {
            var estimator = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(BinaryData.SentimentText))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var newModel = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            mlContext = context;
            model = newModel;

            return newModel;
        }

        public void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            SaveModelAsFile(mlContext, schema, model);
        }

        public void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public Boolean ModelExists()
        {
            ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            if (loadedModel != null)
            {
                return true;
            }

            else return false;
        }

        public void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<BinaryData, BinaryPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<BinaryData, BinaryPrediction>(model);

            BinaryData sampleStatement = new BinaryData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<BinaryData> sentiments = new[]
            {
                new BinaryData
                {
                    SentimentText = "This was a horrible meal"
                },
                new BinaryData
                {
                    SentimentText = "the service was bad but the meal was good"
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<BinaryPrediction> predictedResults = mlContext.Data.CreateEnumerable<BinaryPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (BinaryPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
