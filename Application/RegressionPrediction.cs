using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML;
using System.Linq;
using DataSet;

namespace RegressionPrediction
{

    class RegressionPredict
    {
        [ColumnName("Score")]
        public float FareAmount;
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "prediction-data-test.csv");

        public ITransformer Train(MLContext mlContext, string dataPath)
        {

            IDataView dataView = mlContext.Data.LoadFromTextFile<RegressionData>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return model;
        }

        public void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<RegressionData>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        public void TestSinglePrediction(MLContext mlContext, ITransformer model, RegressionData dataSample)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPredict>(model);

            var prediction = predictionFunction.Predict(dataSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        public void Predict(MLContext mlContext, ITransformer model, RegressionData dataSample)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPredict>(model);

            var prediction = predictionFunction.Predict(dataSample);

            Console.WriteLine($"**********************************************************************");
            // Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
