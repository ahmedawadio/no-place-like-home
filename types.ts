export type MetroDetail = {
  mid: string;
  name: string;
};

export type MetroMetric = {
  mid: string;
  value: number;
  variable_code: string;
  year: number;
};

export type ZipcodeDetails = {
  mid: string;
  zipcode: string;
  city: string;
  state: string;
};

// Define a type that links metro ID to an array of metrics
export type MetroMetricsGroup = {
  metro_id: string;
  metrics: MetroMetric[];
};

export type Variable = {
  variable_code: string;
  name: string;
  description: string;
  type: 'percent' | 'count' | 'dollars'; 
};

export type LocationData = {
  initial_zipcode: string;
  initial_zipcode_found: boolean;  
  zipcode: ZipcodeDetails;
  metro_details: MetroDetail[];
  metro_metrics: MetroMetricsGroup[];  // Array of metro metrics groups
  variables: Variable[];  // Array of variables


};
