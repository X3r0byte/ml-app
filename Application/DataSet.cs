using Microsoft.ML.Data;
using System;

namespace DataSet
{
    /// <summary>
    /// Strongly typed classes for data input into the ML framework
    /// TODO: auto generate from selected dataset
    /// </summary>
    class RegressionData
    {
		[LoadColumn(0)]
		public string VendorId;

		[LoadColumn(1)]
		public string RateCode;

		[LoadColumn(2)]
		public float PassengerCount;

		[LoadColumn(3)]
		public float TripTime;

		[LoadColumn(4)]
		public float TripDistance;

		[LoadColumn(5)]
		public string PaymentType;

		[LoadColumn(6)]
		public float FareAmount;

		[LoadColumn(0)]
		public string Date;

		//[LoadColumn(0)]
		//public int obs;

		//[LoadColumn(1)]
		//public float x1;

		//[LoadColumn(2)]
		//public float x2;

		//[LoadColumn(3)]
		//public float y1;

		//[LoadColumn(4)]
		//public float y2;

		//[LoadColumn(5)]
		//public float y3;

		//[LoadColumn(6)]
		//public float y4;

		//[LoadColumn(6)]
		//public float y5;
	}

    public class RegressionPrediction
    {
        [ColumnName("Score")]
        public float Result;
    }

    class AnomalyData
    {
        [LoadColumn(0)]
        public string time;

        [LoadColumn(1)]
        public float value;
    }

    public class BinaryData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }

}
