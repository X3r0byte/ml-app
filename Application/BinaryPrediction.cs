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
        public static ITransformer trainedModel;
        public static DataViewSchema schema;

        public TrainTestData LoadData(MLContext context, string data)
        {
            mlContext = context;
            IDataView dataView = mlContext.Data.LoadFromTextFile<BinaryData>(data, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            schema = dataView.Schema;
            return splitDataView;
        }

        public void BuildAndTrainModel(IDataView splitTrainSet, bool forceRetrain)
        {
            if(LoadModel() && !forceRetrain)
            {
                return;
            }
            
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(BinaryData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var newModel = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            trainedModel = newModel;
        }

        public void Evaluate(IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = trainedModel.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            SaveModelAsFile(schema);
        }

        public void SaveModelAsFile(DataViewSchema trainingDataViewSchema)
        {
            mlContext.Model.Save(trainedModel, trainingDataViewSchema, modelPath);
        }

        public Boolean LoadModel()
        {
            try
            {
                trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
                return true;
            }
            catch (FileNotFoundException)
            {
                return false;
            }
        }

        public void TestModel()
        {
            IEnumerable<BinaryData> sentiments = new[]
            {
                new BinaryData
                {
                    SentimentText = "this is awesome"
                },
                new BinaryData
                {
                    SentimentText = "can you do this for me?"
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = trainedModel.Transform(batchComments);

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
