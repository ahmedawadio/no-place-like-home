"use client"
import { Input } from '@/components/Input';
import { motion } from "framer-motion";
import { BackgroundBeams } from '@/components/ui/background-beams';
import { MultiStepLoader } from '@/components/ui/multi-step-loader';
import { Header } from '@/components/ui/Header';
import AnalyticsSection from '@/components/section/AnalyticsSection';
import AnalyticsSectionLoader from '@/components/section/AnalyticsSectionLoader';
import { useEffect, useState } from 'react';
import { LocationData } from '@/types';


export default function Home() {

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
    {
      text: "Analyzing Results",
    },
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



  const constructNextImageUrl = (imageUri:string,width:number) => {
    const encodedUrl = encodeURIComponent(imageUri);
    return `/_next/image?url=${encodedUrl}&w=${width}&q=75`;
    
  };
  // instead of paying for a cdn, doing a cheap hack to preload a bunch of images.
  const imageSizes = [
    { width: 2048 },  // Extra large image size
    { width: 1200 },  // Medium image size    
    { width: 1080 },  // Medium image size
    { width: 1920 },  // Large image size
    { width: 640 },   // Small image size
    { width: 3840 },  // 4K resolution image size
  ];
useEffect(() => {
    const preloadImage = (src:any) => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.as = 'image';
      link.href = src;
      document.head.appendChild(link);
    };

    if (locationData?.metro_details) {
      locationData.metro_details.forEach((metro) => {
        if (metro.image_uri) {
          imageSizes.forEach((size) => {
            const imageUrl = constructNextImageUrl(
              metro.image_uri,
              size.width,
            );
            preloadImage(imageUrl);
          });
        }
      });
    }
  }, [locationData])



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
  

{isSubmitted &&
<div className="pt-2">
{/* <Header/> */}



  {

    !isLoading &&

    locationData.initial_zipcode.length?
                  <AnalyticsSection  locationData={locationData}/>
                  :<AnalyticsSectionLoader locationData={locationData}/>


  }

<div className="absolute z-10">

<MultiStepLoader loop={false} setIsLoading={setIsLoading} loadingStates={loadingStates} loading={isLoading}  duration={duration} />
</div>
</div>
}


{!isSubmitted && 
      <div className="touch-pan-y flex flex-col items-center justify-center min-h-screen">



<BackgroundBeams />

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
    <div className='pb-4'>

<a href="https://www.ahmedawad.io/" target="_blank">

    <h2 className="  items-center justify-center bg-gradient-to-br  from-slate-200 to-slate-400 bg-clip-text  text-transparent text-center  text-7xl ">
      Nest ML
   
    </h2>
</a>
    </div>
<Input setSubmittedInput={setSubmittedInput} setIsSubmitted={setIsSubmitted} setIsLoading={setIsLoading}/>
  </motion.div>

</div>

}


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
