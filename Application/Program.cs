using System;
using System.IO;
using Microsoft.ML;
using DataSet;
using static Microsoft.ML.DataOperationsCatalog;

namespace Application
{
    class Program
    {
        static readonly string anomalyData = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        static readonly string regressionData = Path.Combine(Environment.CurrentDirectory, "Data", "prediction-data-train.csv");
        static readonly string binaryData = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        private static readonly string trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues-train.tsv");

        static RegressionPredict regressionPredict = new RegressionPredict();
        static AnomalyDetect anomalyDetect = new AnomalyDetect();
        static Classification classification = new Classification();
        static BinaryPrediction binaryPredict = new BinaryPrediction();

        static MLContext mlContext = new MLContext();

        //assign the Number of records in dataset file to constant variable
        const int _docsize = 20;

        static void Main(string[] args)
        {
            char exit = ' ';
            string input = "";

            Classification.mlContext = mlContext;

            Console.Write("initializing... ");

            InitRegressionPrediction();
            InitBinaryPrediction();
            InitClassification();

            RunAnomalyDetection();

            while (exit != 'q')
            {
                input = Console.ReadLine();
                exit = input[0];
            }
        }

        public static void InitBinaryPrediction()
        {
            TrainTestData splitDataView = binaryPredict.LoadData(mlContext, binaryData);
            ITransformer model = binaryPredict.BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            binaryPredict.Evaluate(mlContext, model, splitDataView.TestSet);
            binaryPredict.UseModelWithBatchItems(mlContext, model);
        }

        public static void InitClassification()
        {
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);
            var pipeline = classification.ProcessData();
            var trainingPipeline = classification.BuildAndTrainModel(trainingDataView, pipeline);

            classification.Evaluate(trainingDataView.Schema);
            classification.PredictIssue();
        }

        public static void InitRegressionPrediction()
        {
            ITransformer regressionModel = regressionPredict.Train(mlContext, regressionData);

            regressionPredict.Evaluate(mlContext, regressionModel);
            regressionPredict.TestSinglePrediction(mlContext, regressionModel);
        }

        public static void RunAnomalyDetection()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<AnomalyData>(path: anomalyData, hasHeader: true, separatorChar: ',');

            anomalyDetect.DetectSpike(mlContext, _docsize, dataView);
            anomalyDetect.DetectChangepoint(mlContext, _docsize, dataView);
        }
    }
}
