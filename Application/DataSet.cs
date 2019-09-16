using Microsoft.ML.Data;

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
    }

    class AnomalyData
    {
        [LoadColumn(2)]
        public string time;

        [LoadColumn(5)]
        public float value;
    }
}
