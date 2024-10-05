type MetroDetail = {
    mid: string;
    name: string;
  };
  
  type MetroMetric = {
    mid: string;
    value: number;
    variable_code: string;
    year: number;
  };
  
  type ZipcodeDetails = {
    mid: string;
    zipcode: string;
    name: string;
  };
  
  type LocationData = {
    has_error: boolean;
    initial_zipcode: string;
    zipcode: ZipcodeDetails;
    metro_details: MetroDetail[][]; // 2D array of MetroDetails, each mid has multiple records
    metro_metrics: (string | MetroMetric[])[];  // Can be a mix of strings and an array of metrics
  };
  