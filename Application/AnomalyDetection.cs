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
            List<int> values = new List<int>();

            var dt = ConvertCSVtoDataTable(data);

            foreach(DataRow row in dt.Rows)
			{
                values.Add(int.Parse(Math.Round(double.Parse(row["value"].ToString()), 0).ToString()));
			}

            var stats = new Statistics();
            var regressionLine = stats.CalculateLinearRegression(values.ToArray());
            var intercepts = regressionLine.xAxisIntercepts.AsQueryable();
            string str = String.Join(",", intercepts);
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

    // https://stackoverflow.com/questions/43224/how-do-i-calculate-a-trendline-for-a-graph/14235891#14235891
    public class Statistics
    {
        public Trendline CalculateLinearRegression(int[] values)
        {
            var yAxisValues = new List<int>();
            var xAxisValues = new List<int>();

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
        private readonly IList<int> xAxisValues;
        private readonly IList<int> yAxisValues;
        public int[] xAxisIntercepts;
        private int count;
        private int xAxisValuesSum;
        private int xxSum;
        private int xySum;
        private int yAxisValuesSum;

        public Trendline(IList<int> yAxisValues, IList<int> xAxisValues)
        {
            this.yAxisValues = yAxisValues;
            this.xAxisValues = xAxisValues;

            this.Initialize();
        }

        public int Slope { get; private set; }
        public int Intercept { get; private set; }
        public int Start { get; private set; }
        public int End { get; private set; }

        private void Initialize()
        {
            this.count = this.yAxisValues.Count;
            this.yAxisValuesSum = this.yAxisValues.Sum();
            this.xAxisValuesSum = this.xAxisValues.Sum();
            this.xxSum = 0;
            this.xySum = 0;

            var intercepts = new List<int>();

            for (int i = 0; i < this.count; i++)
            {
                this.xySum += (this.xAxisValues[i] * this.yAxisValues[i]);
                this.xxSum += (this.xAxisValues[i] * this.xAxisValues[i]);
            }

            this.Slope = this.CalculateSlope();
            this.Intercept = this.CalculateIntercept();
            this.Start = this.CalculateStart();
            this.End = this.CalculateEnd();

            // track the intercepts for y at each x
            for (int i = 0; i < this.xAxisValues.Count; i++)
            {
                var xIntercept = (this.Slope * this.xAxisValues[i]) + this.Intercept;
                intercepts.Add(xIntercept);
            }

            xAxisIntercepts = intercepts.ToArray();
        }

        private int CalculateSlope()
        {
            try
            {
                return ((this.count * this.xySum) - (this.xAxisValuesSum * this.yAxisValuesSum)) / ((this.count * this.xxSum) - (this.xAxisValuesSum * this.xAxisValuesSum));
            }
            catch (DivideByZeroException)
            {
                return 0;
            }
        }

        private int CalculateIntercept()
        {
            return (this.yAxisValuesSum - (this.Slope * this.xAxisValuesSum)) / this.count;
        }

        private int CalculateStart()
        {
            return (this.Slope * this.xAxisValues.First()) + this.Intercept;
        }

        private int CalculateEnd()
        {
            return (this.Slope * this.xAxisValues.Last()) + this.Intercept;
        }
    }
}
