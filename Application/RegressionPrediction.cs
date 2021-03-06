﻿using System;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML;
using System.Linq;
using DataSet;

namespace Application
{
    class RegressionPredict
    {
        public static MLContext mlContext { get; set; }

        //static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "regrex3large.csv");
        // static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model-regression.zip");

        private static string trainPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static string testPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

        public static ITransformer trainedModel;
        public static DataViewSchema schema;
        public static IDataView dataView;

        public void LoadData(MLContext context, string data)
        {
            mlContext = context;
            // dataPath = data;
            dataView = context.Data.LoadFromTextFile<RegressionData>(trainPath, hasHeader: true, separatorChar: ',');
        }

        public void BuildAndTrainModel()
        {
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
				.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
				.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
				.Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
				.Append(mlContext.Regression.Trainers.FastTree());

            var newModel = pipeline.Fit(dataView);

            trainedModel = newModel;
            schema = dataView.Schema;
        }

        public void Evaluate()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<RegressionData>(testPath, hasHeader: true, separatorChar: ',');
            var predictions = trainedModel.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

            SaveModelAsFile(schema);
        }

        public static void SaveModelAsFile(DataViewSchema trainingDataViewSchema)
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

        public void TestModel(RegressionData sample)
        {
           // var taxiTripSample = new RegressionData()
   //         {
			//	Date = ("2/01/2023"),
			//	VendorId = "VTS",
			//	RateCode = "1",
			//	PassengerCount = 1,
			//	TripTime = 1140,
			//	TripDistance = tripDistance,
			//	PaymentType = "CRD",
			//	FareAmount = 0 // To predict. Actual/Observed = 15.5
			//};

            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPrediction>(trainedModel);

            var prediction = predictionFunction.Predict(sample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Prediction: {prediction.Result:0.####}");
            Console.WriteLine($"**********************************************************************");
        }

        public void Predict(MLContext mlContext, ITransformer model, RegressionData dataSample)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPrediction>(model);

            var prediction = predictionFunction.Predict(dataSample);

            Console.WriteLine($"**********************************************************************");
            // Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"Prediction: {prediction.Result:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
