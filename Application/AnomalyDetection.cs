using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using DataSet;

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
        public void DetectChangepoint(MLContext mlContext, int docSize, IDataView data)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(PredictionVector.prediction),
                                                                                    inputColumnName: nameof(AnomalyData.value),
                                                                                    confidence: 95,
                                                                                    changeHistoryLength: docSize / 4);

            ITransformer iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidChangePointTransform.Transform(data);

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

        public void DetectSpike(MLContext mlContext, int docSize, IDataView data)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(PredictionVector.prediction),
                                                                        inputColumnName: nameof(AnomalyData.value),
                                                                        confidence: 95,
                                                                        pvalueHistoryLength: docSize / 4);

            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidSpikeTransform.Transform(data);

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

        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<AnomalyData> enumerableData = new List<AnomalyData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }
    }
}
