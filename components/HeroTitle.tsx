import { motion } from "framer-motion";
import React from "react";


export function HeroTitle() {
  return (
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

{/* <div className="flex flex-col items-center justify-center absolute w-full bottom-0 inset-x-0 h-40 bg-gradient-to-b pointer-events-none select-none from-transparent to-black" > */}

    <h2 className="flex px-4  flex-col items-center justify-center  w-full bg-gradient-to-br absolute top-0 from-slate-200 to-slate-400 py-2 bg-clip-text tracking-tight text-transparent md:text-7x
mt-10   mx-auto  text-center   text-5xl sm:text-6xl  md:text-7xl   ">
      No place like home...kinda
    </h2>



  </motion.div>
  );
}