using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using DataSet;


namespace Application
{
    class Program
    {
        static readonly string anomalyData = Path.Combine(Environment.CurrentDirectory, "Data", "anom.csv");
        static readonly string regressionData = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
       // static readonly string regressionData = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
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
            classification.mlContext = mlContext;

            Console.Write("initializing... ");

            // NetLink.StartServer();

            // InitRegressionPrediction();
            // InitBinaryPrediction();
            // InitClassification();
            RunAnomalyDetection();

            Console.ReadLine();

            //while(true)
            //{
            //    float distance = float.Parse(Console.ReadLine().ToString());
            //    regressionPredict.TestModel(distance);
            //}

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
            regressionPredict.BuildAndTrainModel();

            // evaluate the model against test data
            regressionPredict.Evaluate();

			// test output with a test method

			var sample = 
			new RegressionData {
				VendorId = "VTS",
				RateCode = "1",
				PassengerCount = 1,
				TripTime = 900,
				TripDistance = 8.63f,
				PaymentType = "CRD",
				FareAmount = 0 // To predict. Actual/Observed = 15.5
			};
			regressionPredict.TestModel(sample);
        }

        public static void RunAnomalyDetection()
        {
            // load the mlContext and data path
            anomalyDetect.LoadData(mlContext, anomalyData);

            // anomaly detection does not require a model

            // evaluate model against test data, detect a spike
            anomalyDetect.DetectSpike(_docsize);

            // evaluate model against test data, detect a changepoint
            //anomalyDetect.DetectChangepoint(_docsize);
        }
    }



}
