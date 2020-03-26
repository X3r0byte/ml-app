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

        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "regrex3large.csv");
        // static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
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
            //if (LoadModel() && !forceRetrain)
            //{
            //    return;
            //}

            var pipeline = mlContext.Transforms.CopyColumns(
                outputColumnName: "Label", inputColumnName: "y5")
                //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DateEncoded", inputColumnName: "Date"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "obsEncoded", inputColumnName: "obs"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "x1Encoded", inputColumnName: "x1"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "x2Encoded", inputColumnName: "x2"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "y1Encoded", inputColumnName: "y1"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "y2Encoded", inputColumnName: "y2"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "y3Encoded", inputColumnName: "y3"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "y4Encoded", inputColumnName: "y4"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "y5Encoded", inputColumnName: "y5"))

                .Append(mlContext.Transforms.Concatenate("Features", "obsEncoded", "x1Encoded", "x2Encoded", "y1Encoded", "y2Encoded", "y3Encoded", "y4Encoded"))

                //outputColumnName: "Label", inputColumnName: "FareAmount")
                //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                //.Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))

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

        public void TestModel(float tripDistance)
        {
            var taxiTripSample = new RegressionData()
            {
               // Date = ("2/01/2023"),
               // Billable = 60
                //VendorId = "VTS",
                //RateCode = "1",
                //PassengerCount = 1,
                //TripTime = 1140,
                //TripDistance = tripDistance,
                //PaymentType = "CRD",
                //FareAmount = 0 // To predict. Actual/Observed = 15.5


                obs = 20,
                x1 = 21.9663f,
                x2 = 34.9375f,
                y1 = 18.9276f,
                y2 = 18.5008f,
                y3 = 18.9919f,
                y4 = 18.9195f
                //y5 = 93.0904f



            };

            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPrediction>(trainedModel);

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Prediction: {prediction.y5:0.####}");
            Console.WriteLine($"**********************************************************************");
        }

        public void Predict(MLContext mlContext, ITransformer model, RegressionData dataSample)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<RegressionData, RegressionPrediction>(model);

            var prediction = predictionFunction.Predict(dataSample);

            Console.WriteLine($"**********************************************************************");
            // Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"Prediction: {prediction.y5:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
