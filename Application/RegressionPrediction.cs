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

        public static ITransformer trainedModel;
        public static DataViewSchema schema;
        public static IDataView dataView;

        public void LoadData(MLContext context, string data)
        {
            mlContext = context;
            dataView = context.Data.LoadFromTextFile<RegressionData>(data, hasHeader: true, separatorChar: ',');
        }

        public void BuildAndTrainModel(bool forceRetrain)
        {
            if(LoadModel() && !forceRetrain)
            {
                return;
            }

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var newModel = pipeline.Fit(dataView);

            trainedModel = newModel;
            schema = dataView.Schema;
        }

        public void Evaluate()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<RegressionData>(_testDataPath, hasHeader: true, separatorChar: ',');
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

        public void TestModel()
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

            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPredict>(trainedModel);

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
