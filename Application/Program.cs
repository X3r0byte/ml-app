using System;
using System.IO;
using Microsoft.ML;
using AnomalyDetection;
using RegressionPrediction;
using DataSet;

namespace Application
{
    class Program
    {
        static readonly string anomalyData = Path.Combine(Environment.CurrentDirectory, "Data", "anomaly-data.csv");
        static readonly string regressionData = Path.Combine(Environment.CurrentDirectory, "Data", "prediction-data-train.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static ITransformer regressionModel;
        static RegressionPredict regressionPredict = new RegressionPredict();
        static AnomalyDetect anomalyDetect = new AnomalyDetect();

        static MLContext mlContext = new MLContext();

        //assign the Number of records in dataset file to constant variable
        const int _docsize = 213;

        static void Main(string[] args)
        {
            char exit = ' ';
            string input = "";

            Console.Write("initializing... ");

            InitRegressionPrediction();
            // RunAnomalyDetection();

            while(exit != 'q')
            {
                if(input == "test")
                {
                    Predict();
                }

                // test


                input = Console.ReadLine();
                exit = input[0];
            }
        }

        public static void InitRegressionPrediction()
        {
            regressionModel = regressionPredict.Train(mlContext, regressionData);
            regressionPredict.Evaluate(mlContext, regressionModel);

            var taxiTripSample = new RegressionData()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            regressionPredict.TestSinglePrediction(mlContext, regressionModel, taxiTripSample);
        }

        public static void Predict()
        {
            var taxiTripSample = new RegressionData()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 2,
                TripTime = 1620,
                TripDistance = 2.79f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict.
            };

            regressionPredict.Predict(mlContext, regressionModel, taxiTripSample);
        }

        public static void RunAnomalyDetection()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<AnomalyDetect>(path: anomalyData, hasHeader: true, separatorChar: ',');
            anomalyDetect.DetectSpike(mlContext, _docsize, dataView);
            anomalyDetect.DetectChangepoint(mlContext, _docsize, dataView);
        }
    }
}
