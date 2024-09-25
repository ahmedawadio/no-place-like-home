import Image from 'next/image'
import Link from 'next/link'
import{FaExclamationTriangle} from 'react-icons/fa'

export default function Home() {
  return (
    <div className="">

<div className="overflow-visible w-full h-full pb-40  ">


{/* <Spotlight className=" pl-50 ml-10 overflow-visible"/> */}
<div className="  w-screen 
px-2 py-10 ">  

<div className="
 bg-gradient-to-br from-slate-300 to-slate-500 py-4 bg-clip-text tracking-tight text-transparent md:text-7x

ml-10 mt-20 text-6xl md:text-7xl sm:text-6xl xs:text-2xl  mx-auto font-normal text-left text-neutral-200">

  Nest <br />
  No place like home..
 
</div>
<StillUnderConstruction/>
</div>

  </div>
   
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
