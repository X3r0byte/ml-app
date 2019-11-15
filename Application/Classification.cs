using System;
using System.IO;
using Microsoft.ML;
using DataSet;
using System.Linq;

namespace Application
{
    public class Classification
    {
        public  MLContext mlContext { get; set; }

        private static readonly string appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static readonly string testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues-test.tsv");
        private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model-classification.zip");

        public static PredictionEngine<GitHubIssue, IssuePrediction> predEngine;
        public static ITransformer trainedModel;
        public static IDataView trainingDataView;

        public void LoadData(MLContext context, string data)
        {
            mlContext = context;
            trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(data, hasHeader: true);
        }

        public IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            return pipeline;
        }

        public void BuildAndTrainModel(IEstimator<ITransformer> pipeline, bool forceRetrain)
        {
            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                                           .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            if (LoadModel() && !forceRetrain)
            {
                return;
            }

            trainedModel = trainingPipeline.Fit(trainingDataView);
            predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

        }

        public void Evaluate()
        {
            var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(trainingDataView.Schema);
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
            catch(FileNotFoundException)
            {
                return false;
            }
        }

        public void TestModel()
        {
            GitHubIssue singleIssue = new GitHubIssue()
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing"
            };

            predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);
            var prediction = predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
    }
}
