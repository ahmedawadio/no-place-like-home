"use client"
// // import { Globe } from '@/components/ui/globe'
import { Input } from '@/components/Input';
import { SpinningGlobe } from '@/components/SpinningGlobe';
// // import { GlobeDemo } from '@/components/world';
import { motion } from "framer-motion";
import { BackgroundBeams } from '@/components/ui/background-beams';
import { HeroTitle } from '@/components/HeroTitle';
// // import { LoaderOverlay } from '@/components/LoaderOverlay';
import { MultiStepLoader } from '@/components/ui/multi-step-loader';
import { Header } from '@/components/ui/Header';
import AnalyticsSection from '@/components/section/AnalyticsSection';
import AnalyticsSectionLoader from '@/components/section/AnalyticsSectionLoader';


// import{FaExclamationTriangle} from 'react-icons/fa'
import { useEffect, useState } from 'react';
import { LocationData, MetroDetail } from '@/types';
import { World } from '@/components/ui/globe';
import { UsaGlobe } from '@/components/UsaGlobe';

export default function Home() {

// return<div></div>
  const loadingStates = [
    {
      text: "Conencting to API",
    },
    // {
    //   text: "Querying Database",
    // },
    {
      text: "Running Model",
    },
    // {
    //   text: "Analyzing Results",
    // },
    {
      text: "Clicking Heels 3x",
    },
    // {
    //   text: "Grabbing Coffee",
    // },
    {
      text: "Generating Report",
    },
  ];
  
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [submittedInput, setSubmittedInput] = useState("");


  const duration = 2000

  const defaultLocationData: LocationData = {
    initial_zipcode: "",
    initial_zipcode_found: false,
    zipcode: {
      mid: "",
      zipcode: "",
      city: "",
      state: "",
    },
    metro_details: [], 
    metro_metrics: [],
    variables: [],
  };
  
  const [locationData, setLocationData] = useState<LocationData>(defaultLocationData);

  useEffect(() => {
    // Fetch data from the Flask API
    if (submittedInput) {

    const fetchLocationData = async () => {
      try {
        const response = await fetch(`/api/zipcode/${submittedInput}`);
        const data = await response.json();
        console.log(data.has_error)
        setLocationData(data); // Set the response data into state

      } catch (error) {
        
        console.error({error});
      }
    };

    fetchLocationData();
  }
  }, [submittedInput]);


  useEffect(() => {
    
    const preloadImage = (src:any) => {
      const link = document.createElement("link");
      link.rel = "preload";
      link.as = "image";
      link.href = src;
      document.head.appendChild(link);
    };
  
    if (locationData.metro_details) {
      locationData.metro_details.forEach((metro) => {
        if (metro.image_uri) {
          preloadImage(metro.image_uri);
        }
      });
    }
  }, [locationData]);

  const isTest = false

  useEffect(() => {
    if (isTest){
      setIsSubmitted(true)
      setSubmittedInput("12345");
    }
  }, [setSubmittedInput]);


  // return(null)


  return (
  
  <>
  
      <Header/>
    <div className="pt-2">

{isSubmitted &&
<>


  {

    !isLoading &&

    locationData.initial_zipcode.length?
                  <AnalyticsSection  locationData={locationData}/>
                  :<AnalyticsSectionLoader locationData={locationData}/>


  }

<div className="absolute z-10">

<MultiStepLoader loop={false} setIsLoading={setIsLoading} loadingStates={loadingStates} loading={isLoading}  duration={duration} />
</div>
</>
}

{!isSubmitted && 
<>



<BackgroundBeams />
<HeroTitle/>
 {/* <SpinningGlobe/>  */}

<UsaGlobe/>
<motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 1,
          }}
          className="div"
        >

      <div className="absolute w-full bottom-0 inset-x-0 h-80 bg-gradient-to-b  from-transparent to-black" />
      <div className="flex flex-col items-center justify-center absolute w-full bottom-0 inset-x-0 h-40 bg-gradient-to-b  from-transparent to-black" >


             <Input setSubmittedInput={setSubmittedInput} setIsSubmitted={setIsSubmitted} setIsLoading={setIsLoading}/>


   
</div>
</motion.div>


</>
}


  </div>
</>

   
  )
}

// function StillUnderConstruction() {

//   return(

//     <div className="flex ml-10 mt-8">
//     <div className="flex items-center border border-yellow-500 bg-yellow-200 bg-opacity-10 text-yellow-300 text-md font-semibold px-4 py-2 rounded-2xl">
//       <FaExclamationTriangle className="w-5 h-5 mr-2 text-yellow-300" />
//       Under Construction
//     </div>
//   </div>
//   )
//  }
