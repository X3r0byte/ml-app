using System;
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

        [ColumnName("Score")]
        public float FareAmount;
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "prediction-data-test.csv");
        private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model-regression.zip");

        ITransformer model;
        DataViewSchema schema;

        public ITransformer Train(MLContext context, string dataPath)
        {
            IDataView dataView = context.Data.LoadFromTextFile<RegressionData>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(context.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(context.Regression.Trainers.FastTree());

            var newModel = pipeline.Fit(dataView);

            model = newModel;
            mlContext = context;

            schema = dataView.Schema;

            return newModel;
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

            SaveModelAsFile(mlContext, schema, model);
        }

        public static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public static Boolean ModelExists()
        {
            ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            if (loadedModel != null)
            {
                return true;
            }

            else return false;
        }

        public void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
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

            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPredict>(model);

            var prediction = predictionFunction.Predict(taxiTripSample);

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
