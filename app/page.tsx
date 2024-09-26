"use client"
// import { Globe } from '@/components/ui/globe'
import { Input } from '@/components/Input';
import { SpinningGlobe } from '@/components/SpinningGlobe';
// import { GlobeDemo } from '@/components/world';
import Image from 'next/image'
import Link from 'next/link'
import{FaExclamationTriangle} from 'react-icons/fa'
import { motion } from "framer-motion";
import { BackgroundBeams } from '@/components/ui/background-beams';
import { HeroTitle } from '@/components/HeroTitle';
import { LoaderOverlay } from '@/components/LoaderOverlay';
import { useState } from 'react';
import { MultiStepLoader } from '@/components/ui/multi-step-loader';

export default function Home() {

  const loadingStates = [
    {
      text: "Querying backend",
    },
    {
      text: "Running ML model",
    },
    {
      text: "Finding home cluster",
    },
    {
      text: "Organizing results",
    },
    {
      text: "Clicking heels 3 times",
    },
    {
      text: "Grabbing coffee",
    },
    {
      text: "Sending results",
    },
  ];
  
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const duration = 2000
  

  
  return (
    <div className="">

{/* <div className="pacifico-regular mt-10 ml-10 text-2xl absolute text-white text-opacity-20">
  <p className="custom-text-outline">Nest</p>
</div> */}



{isSubmitted &&
<>
  {

    !isLoading &&

<div className="bg-black h-screen w-screen flex items-center justify-center">
  <img
    src="https://wallpapers.com/images/hd/that-s-all-folks-dark-theme-fvjibwtktkmmxrhc.jpg"
    alt="That's All Folks"
    className="max-w-full max-h-full object-contain"
  />
  <p className="absolute bottom-20 text-white text-5xl font-bold">🐷 lol</p>
  </div>
  }
<div className="absolute z-10">

<MultiStepLoader loop={false} setIsLoading={setIsLoading} loadingStates={loadingStates} loading={isLoading}  duration={duration} />
</div>
</>
}
{!isSubmitted && 

<>

<div className="absolute z-10">

<MultiStepLoader loop={false} setIsLoading={setIsLoading} loadingStates={loadingStates} loading={isLoading}  duration={duration} />
</div>


{/* <Input/> */}
<BackgroundBeams />
<HeroTitle/>



{!isSubmitted &&

<SpinningGlobe/>
}

{/* <LoaderOverlay/> */}


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

{/* <div className="mb-20 h-10 w-500 mx-auto p-4 flex flex-col items-center justify-center"> */}

             <Input setIsSubmitted={setIsSubmitted} setIsLoading={setIsLoading}/>


          <p className="text-center  w-full text-md font-normal text-neutral-400 dark:text-neutral-100 max-w-md  ">
            Enter your zipcode to find homes away from home
          </p>
</div>
</motion.div>
</>

}
</div>

   
  )
}

function StillUnderConstruction() {

  return(

    <div className="flex ml-10 mt-8">
    <div className="flex items-center border border-yellow-500 bg-yellow-200 bg-opacity-10 text-yellow-300 text-md font-semibold px-4 py-2 rounded-2xl">
      <FaExclamationTriangle className="w-5 h-5 mr-2 text-yellow-300" />
      Under Construction
    </div>
  </div>
  )
 }
