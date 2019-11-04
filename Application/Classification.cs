using System;
using System.IO;
using Microsoft.ML;
using DataSet;
using System.Linq;

namespace Application
{
    public class Classification
    {
        public static MLContext mlContext { get; set; }

        private static readonly string appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static readonly string testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues-test.tsv");
        private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model-classification.zip");

        private static PredictionEngine<GitHubIssue, IssuePrediction> predEngine;
        private static ITransformer trainedModel;
        static IDataView _trainingDataView;

        public IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            return pipeline;
        }

        public IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                                           .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            trainedModel = trainingPipeline.Fit(trainingDataView);

            predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        public void Evaluate(DataViewSchema trainingDataViewSchema)
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

            SaveModelAsFile(mlContext, trainingDataViewSchema, trainedModel);
        }

        public void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public Boolean ModelExists()
        {
            ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            if (loadedModel != null)
            {
                return true;
            }

            else return false;
        }

        public void PredictIssue()
        {
            ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue()
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing"
            };

            predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
            var prediction = predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
    }
}
