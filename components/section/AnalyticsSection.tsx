import React, { useEffect, useState } from "react";
import { MetroComboboxPopover } from "../MetroComboboxPopover";
import { LocationData, MetroDetail, Variable } from "@/types";
import VariablesGraph from "../VariablesGraph";
import { VariableComboboxPopover } from "../ui/VariableComboboxPopover";

interface Props {
  locationData: LocationData;
  setLocationData: React.Dispatch<React.SetStateAction<LocationData>>;
}

export default function AnalyticsSection({
  locationData,
  setLocationData,
}: Props) {
 
  const metroDetails = locationData.metro_details;
  const similarMetroDetails = locationData.metro_details.slice(1);
  const [selectedMetro, setSelectedMetro] = useState<MetroDetail>(similarMetroDetails[0]);
  const [selectedVariable, setSelectedVariable] = useState<Variable>(locationData.variables[0]);


  return (
<div className="bg-transparent my-20 w-full max-w-[calc(100%-40px)] mx-auto flex flex-col gap-6 justify-start overflow-y-auto py-6">
<div className=" flex flex-col gap-6 items-start justify-start p2-">
        <div
          id="header"
          className="border border-transparent rounded-lg p-4 w-full  "
        >
          <h2 className="  text-3xl font-semibold tracking-tight  first:mt-0">
            {locationData.metro_details[0]["name"]}
          </h2>

          <p className="mt-2 opacity-50">
            {locationData.zipcode.zipcode} {locationData.zipcode.city},{" "}
            {locationData.zipcode.state}
          </p>
        </div>

        <div id="home away from home" className="">
          {/* <h3 className=" text-3xl font-semibold tracking-tight  first:mt-0">
       {locationData.metro_details[1]["name"]}

      </h3>

    
      <p className="mt-2 opacity-50" >

      home away from home
      </p> */}
        </div>
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
  );
}
