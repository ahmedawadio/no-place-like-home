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

  return (

<>
    <div className="bg-transparent pl-20 pr-36  mt-20 w-full max-w-[calc(100%-40px)] mx-auto flex flex-col gap-6 justify-start  overflow-y-clip py-6">
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
              {`You entered: `}<code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-3xl font-semibold">{locationData.initial_zipcode}</code>
            </h2>
            <p className="leading-7 [&:not(:first-child)]:mt-6">
              {locationData.initial_zipcode_found ? (
                `The zipcode ${locationData.zipcode.zipcode} is part of the metro area ${locationData.metro_details[0].name}, located in ${locationData.zipcode.city}, ${locationData.zipcode.state}.`
              ) : (
                <>We could not find <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-3xl font-semibold">{locationData.initial_zipcode}</code>. We matched the closest zipcode: {locationData.zipcode.zipcode}, which is part of {locationData.zipcode.city}, {locationData.zipcode.state}.</>
              )}
            </p>
            <h3 className="mt-8 border-b pb-2 text-3xl font-semibold tracking-tight w-full">
              About Your Area
            </h3>
            <p className="leading-7 [&:not(:first-child)]:mt-6">
              The area you selected is rich in culture and diversity, offering a variety of amenities and opportunities for residents. Whether you enjoy outdoor activities, dining experiences, or cultural events, there's something for everyone.
            </p>
          </div>


          <h2 className="mt-10  border-b pb-2 text-3xl font-semibold tracking-tight transition-colors first:mt-0 w-full">
          Model Results</h2>
          <p className="leading-7 [&:not(:first-child)]:mt-6">
            These are the top {similarMetroDetails.length} similar metropolitan areas:
          </p>
          <ol className="my-6 ml-6 list-decimal [&>li]:mt-2">
            {similarMetroDetails.map((metro) => (
              <li key={metro.mid}>
                <a href={`#metro-${metro.mid}`} className="text-primary underline underline-offset-4">
                  {metro.name}
                </a>
              </li>
            ))}
          </ol>
            <MetroCarousel/>

          {similarMetroDetails.map((metro, index) => (
            <div key={metro.mid} id={`metro-${metro.mid}`} className="mt-8">
              <h4 className=" text-xl font-semibold tracking-tight">
                {`${index + 1}. ${metro.name}`}
              </h4>
              <p className="leading-7 [&:not(:first-child)]:mt-6">
                {`${metro.name} is known for its vibrant community, excellent public services, and a wide range of recreational opportunities. It's a great place for families, young professionals, and retirees alike.`}
              </p>
            </div>
          ))}
          

            <h2 className="mt-10 border-b pb-2 text-3xl font-semibold tracking-tight transition-colors first:mt-0 w-full">
            Key Variables Used in the Analysis
          </h2>
          <p className="leading-7 [&:not(:first-child)]:mt-6">
            The model ran K-means clustering across thousands of variables spanning the past 5 years. Below are the top variables that were used in the model, which include popular variables as well as those that exhibited high variance and were useful for dimensionality reduction.
          </p>
          <ul className="my-6 ml-6 list-disc [&>li]:mt-2">
            {locationData.variables.slice(0, 5).map((variable) => (
              <li key={variable.variable_code}>
                <strong>{variable.name}:</strong> {variable.description}
              </li>
            ))}
          </ul>

          <h3 className="mt-10 border-b pb-2 text-3xl font-semibold tracking-tight w-full">
            Compare Metro Area Variables
          </h3>
          <p className="leading-7 [&:not(:first-child)]:mt-6">
            Below, you can explore your entered zipcode's metro area and compare key variables over the past few years. This comparison showcases which cities excel in different variables, providing insights into the unique characteristics of each area.
          </p>
        </div>


      </TracingBeam>


    </div>
    <div className=" px-36 pb-15 ">

      <GraphSection locationData={locationData} />
    </div>

            </>
  );
}
