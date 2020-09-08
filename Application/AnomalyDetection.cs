using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using DataSet;
using System.Linq;
using System.Data;
using System.IO;
using System.Text.RegularExpressions;

namespace Application
{
    public class PredictionVector
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] prediction { get; set; }
    }

    public class AnomalyDetect
    {
        public static IDataView dataView;
        public static MLContext mlContext;

        public void LoadData(MLContext context, string data)
        {
            mlContext = context;
            dataView = mlContext.Data.LoadFromTextFile<AnomalyData>(path: data, hasHeader: true, separatorChar: ',');
            var values = new List<double>();

            var dt = ConvertCSVtoDataTable(data);

            foreach(DataRow row in dt.Rows)
			{
                values.Add(double.Parse(row["value"].ToString()));
			}

            var stats = new Statistics();
            var regressionLine = stats.CalculateLinearRegression(values.ToArray());
            var intercepts = regressionLine.xAxisIntercepts.AsQueryable();
            string str = String.Join(",\n", intercepts);
        }

        public void DetectChangepoint(int docSize)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(PredictionVector.prediction),
                                                                                    inputColumnName: nameof(AnomalyData.value),
                                                                                    confidence: 95,
                                                                                    changeHistoryLength: docSize / 4);

            ITransformer iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView());
            IDataView transformedData = iidChangePointTransform.Transform(dataView);

            var predictions = mlContext.Data.CreateEnumerable<PredictionVector>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            foreach (var p in predictions)
            {
                var results = $"{p.prediction[0]}\t{p.prediction[1]:f2}\t{p.prediction[2]:F2}\t{p.prediction[3]:F2}";

                if (p.prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }
                Console.WriteLine(results);
            }

            Console.WriteLine("");
        }

        public void DetectSpike(int docSize)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(PredictionVector.prediction),
                                                                        inputColumnName: nameof(AnomalyData.value),
                                                                        confidence: 95,
                                                                        pvalueHistoryLength: docSize / 4);

            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView());
            IDataView transformedData = iidSpikeTransform.Transform(dataView);

            var predictions = mlContext.Data.CreateEnumerable<PredictionVector>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                var results = $"{p.prediction[0]}\t{p.prediction[1]:f2}\t{p.prediction[2]:F2}";

                if (p.prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }

            Console.WriteLine("");
        }

        public IDataView CreateEmptyDataView()
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<AnomalyData> enumerableData = new List<AnomalyData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        public DataTable ConvertCSVtoDataTable(string strFilePath)
        {
            var sr = new StreamReader(strFilePath);
            string[] headers = sr.ReadLine().Split(',');
            var dt = new DataTable();
            foreach (string header in headers)
            {
                dt.Columns.Add(header);
            }
            while (!sr.EndOfStream)
            {
                string[] rows = Regex.Split(sr.ReadLine(), ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                DataRow dr = dt.NewRow();
                for (int i = 0; i < headers.Length; i++)
                {
                    dr[i] = rows[i];
                }
                dt.Rows.Add(dr);
            }
            return dt;
        }

    }

    // ripped and modified from https://stackoverflow.com/questions/43224/how-do-i-calculate-a-trendline-for-a-graph/14235891#14235891
    public class Statistics
    {
        public Trendline CalculateLinearRegression(double[] values)
        {
            var yAxisValues = new List<double>();
            var xAxisValues = new List<double>();

            for (int i = 0; i < values.Length; i++)
            {
                yAxisValues.Add(values[i]);
                xAxisValues.Add(i + 1);
            }

            return new Trendline(yAxisValues, xAxisValues);
        }
    }

    public class Trendline
    {
        private readonly IList<double> xAxisValues;
        private readonly IList<double> yAxisValues;
        public double[] xAxisIntercepts;
        private int count;
        private double xAxisValuesSum;
        private double xxSum;
        private double xySum;
        private double yAxisValuesSum;

        public Trendline(IList<double> yVals, IList<double> xVals)
        {
            yAxisValues = yVals;
            xAxisValues = xVals;

            Initialize();
        }

        public double Slope { get; private set; }
        public double Intercept { get; private set; }
        public double Start { get; private set; }
        public double End { get; private set; }

        private void Initialize()
        {
            count = yAxisValues.Count;
            yAxisValuesSum = yAxisValues.Sum();
            xAxisValuesSum = xAxisValues.Sum();
            xxSum = 0;
            xySum = 0;

            var intercepts = new List<double>();

            for (int i = 0; i < count; i++)
            {
                xySum += (xAxisValues[i] * yAxisValues[i]);
                xxSum += (xAxisValues[i] * xAxisValues[i]);
            }

            Slope = CalculateSlope();
            Intercept = CalculateIntercept();
            Start = CalculateStart();
            End = CalculateEnd();

            // track the intercepts for y at each x
            for (int i = 0; i < xAxisValues.Count; i++)
            {
                var xIntercept = (Slope * xAxisValues[i]) + Intercept;
                intercepts.Add(xIntercept);
            }

            xAxisIntercepts = intercepts.ToArray();
        }

        private double CalculateSlope()
        {
            try
            {
                return ((count * xySum) - (xAxisValuesSum * yAxisValuesSum)) / ((count * xxSum) - (xAxisValuesSum * xAxisValuesSum));
            }
            catch (DivideByZeroException)
            {
                return 0;
            }
        }

        private double CalculateIntercept()
        {
            return (yAxisValuesSum - (Slope * xAxisValuesSum)) / count;
        }

        private double CalculateStart()
        {
            return (Slope * xAxisValues.First()) + Intercept;
        }

        private double CalculateEnd()
        {
            return (Slope * xAxisValues.Last()) + Intercept;
        }
    }
}
