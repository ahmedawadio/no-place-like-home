import React, { useEffect, useState } from "react";
import { MetroComboboxPopover } from "../MetroComboboxPopover";
import { LocationData, MetroDetail, Variable } from "@/types";
import VariablesGraph from "../VariablesGraph";
import { VariableComboboxPopover } from "../VariableComboboxPopover";
import { ListFilter, TrendingUpDown } from "lucide-react";
import GraphSection from "./GraphSection";
import { TracingBeam } from "../ui/tracing-beam";
import { MetroCarousel } from "./MetroCarousel";

interface Props {
  locationData: LocationData;
}

export default function AnalyticsSection({
  locationData,
}: Props) {
 
  const similarMetroDetails = locationData.metro_details.slice(1);

  const renderParagraphs = (text: string) => {
    return text.split('\n\n').map((paragraph, index) => (
      <p key={index} className="mb-4">
        {paragraph}
      </p>
    ));
  };

  return (

    <div className="bg-transparent px-10  overflow-x-hidden  mt-20 w-full max-w-[calc(100%-40px)] mx-auto flex flex-col gap-6 justify-start  overflow-y-clip py-6 pb-20">
          <TracingBeam className="w-full mb-10">
        <div className="flex flex-col gap-6 items-start justify-start">
          <div
            id="header"
            className="border border-transparent rounded-lg w-full"
            >
            <h1 className=" text-4xl font-extrabold tracking-tight lg:text-5xl">
              Discover Your New Community
            </h1>
            <h2 className="mt-10 border-b pb-2 text-3xl font-semibold tracking-tight transition-colors first:mt-0 w-full">
              {`You entered: `}<code className="relative rounded bg-muted px-[0.4rem] py-[0.2rem] font-mono text-3xl font-semibold">{locationData.initial_zipcode}</code>
            </h2>
            <p className="leading-7 mt-6">
  {locationData.initial_zipcode_found ? (
    <>
      The zipcode {locationData.zipcode.zipcode} is part of the {locationData.metro_details[0].name}.
    </>
  ) : (
    <>
      We couldn't find {locationData.initial_zipcode}. The closest match is {locationData.zipcode.zipcode}, which is part of {locationData.zipcode.city}, {locationData.zipcode.state} in the {locationData.metro_details[0].name}.
    </>
  )}
</p>
            {/* <h3 className="mt-8 border-b pb-2 text-3xl font-semibold tracking-tight w-full">
              About Your Area
            </h3> */}
            <p className="leading-7 [&:not(:first-child)]:mt-6">
              {renderParagraphs(locationData.metro_details[0].about).slice(0, 1)}
            </p>
          </div>


          <h2 className="my-5 mb-10  border-b pb-2 text-3xl font-semibold tracking-tight transition-colors first:mt-0 w-full">
          Model Results</h2>
   
        
            <MetroCarousel metro_details={locationData.metro_details.slice(1)}/>

        
          

            <h2 className="mt-10 border-b pb-2 text-3xl font-semibold tracking-tight transition-colors first:mt-0 w-full">
            Key Variables Used in the Analysis
          </h2>
          <p className="leading-7 [&:not(:first-child)]:mt-6">
          The analysis leveraged K-means clustering and nearest neighbors on thousands of features derived from five years of data. Presented below are the primary variables selected by Random Forest for their substantial variance and effectiveness in dimensionality reduction.           </p>
          <ul className="my-6 ml-6 list-disc [&>li]:mt-2">
            {locationData.variables.slice(0, 5).map((variable) => (
              <li key={variable.variable_code}>
                <strong className="text-lg">{variable.name}:</strong> {variable.description}
              </li>
            ))}
          </ul>

          <h3 className="mt-10 border-b pb-2 text-3xl font-semibold tracking-tight w-full">
            Compare Metro Areas
          </h3>
          <p className="leading-7 [&:not(:first-child)]:mt-6 pb-10">
          Explore the {locationData.metro_details[0].name} metro area and compare its key variables with other regions over time to help you find your home away from home.
        </p>
        </div>


      </TracingBeam>

      <div className="px-28">

      <GraphSection locationData={locationData} />
      </div>



    </div>


  );
}
