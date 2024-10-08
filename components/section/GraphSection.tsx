import React, { useEffect, useState } from "react";
import { MetroComboboxPopover } from "../MetroComboboxPopover";
import { LocationData, MetroDetail, Variable } from "@/types";
import VariablesGraph from "../VariablesGraph";
import { VariableComboboxPopover } from "../VariableComboboxPopover";
import { ListFilter, TrendingUpDown } from "lucide-react";

interface Props {
  locationData: LocationData;
}

export default function GraphSection({
  locationData,
}: Props) {
 
  const metroDetails = locationData.metro_details;
  const similarMetroDetails = locationData.metro_details.slice(1);
  const [selectedMetro, setSelectedMetro] = useState<MetroDetail>(similarMetroDetails[0]);
  const [selectedVariable, setSelectedVariable] = useState<Variable>(locationData.variables[0]);


  return (
<div className=" flex flex-col gap-y-4 mt-5 ">

<div className="flex flex-row gap-x-4  items-center can overflow-x-auto max-w-full whitespace-nowrap pb-4 md:pb-0">

<div className="relative px-1 ">
    <h3 className="text-md opacity-50 font-semibold tracking-tight flex items-center">
      <ListFilter className="w-5 h-5 mr-3" />
      <span>Filters</span>
      
    </h3>
    </div>
    <div className="mx-3 h-8 min-w-[1px] w-[1px] bg-white opacity-40"></div>


  
<MetroComboboxPopover
  selectedMetro={selectedMetro}
  setSelectedMetro={setSelectedMetro}
  metroDetails={similarMetroDetails}
  />

<VariableComboboxPopover
  selectedVariable={selectedVariable}
  setSelectedVariable={setSelectedVariable}
  variables={locationData.variables}
  />
  </div>

  <div className="">
      <VariablesGraph
        metroMetrics={locationData.metro_metrics}
        selectedMetro={selectedMetro}
        selectedVariable={selectedVariable.variable_code}
        selectedLocations={[
          locationData.metro_details[0]?.name,
          selectedMetro?.name,
        ]}
        variables={locationData.variables}
        
      />
  </div>
    </div>
    );
  }
