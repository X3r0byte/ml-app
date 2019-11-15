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

            classification.mlContext = mlContext;

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
            // load the mlContext and data path
            TrainTestData splitDataView = binaryPredict.LoadData(mlContext, binaryData);

            // create the model, boolean to force model retrain
            binaryPredict.BuildAndTrainModel(splitDataView.TrainSet, false);

            // evaluate the model against test data
            binaryPredict.Evaluate(splitDataView.TestSet);

            // test output with a test method
            binaryPredict.TestModel();
        }

        public static void InitClassification()
        {
            // load the mlContext and data path
            classification.LoadData(mlContext, trainDataPath);
            IEstimator<ITransformer> pipeline = classification.ProcessData();

            // create the model, boolean to force model retrain
            classification.BuildAndTrainModel(pipeline, false);

            // evaluate the model against test data
            classification.Evaluate();

            // test output with a test method
            classification.TestModel();
        }

        public static void InitRegressionPrediction()
        {
            // load the mlContext and data path
            regressionPredict.LoadData(mlContext, regressionData);

            // create the model, boolean to force model retrain
            regressionPredict.BuildAndTrainModel(false);

            // evaluate the model against test data
            regressionPredict.Evaluate();

            // test output with a test method
            regressionPredict.TestModel();
        }

        public static void RunAnomalyDetection()
        {
            // load the mlContext and data path
            anomalyDetect.LoadData(mlContext, anomalyData);

            // anomaly detection does not require a model

            // evaluate model against test data
            anomalyDetect.DetectSpike(_docsize);
            anomalyDetect.DetectChangepoint(_docsize);
        }
    }
}
