import React, { useEffect } from 'react'


interface Props {
  locationData:LocationData;
  setLocationData: (location: string) => void; 
}


export default function AnalyticsSection({locationData, setLocationData}: Props) {

  // console.log("here",locationData.metro_details[0][0]["name"])
  return (
      <div className="bg-black h-screen w-screen flex items-center justify-center">
  <img
    src="https://wallpapers.com/images/hd/that-s-all-folks-dark-theme-fvjibwtktkmmxrhc.jpg"
    alt="That's All Folks"
    className="max-w-full max-h-full object-contain"
  />
  <p className="absolute bottom-20 text-white text-xs font-bold">
    {/* 🐷 still working on the rest lol */}
    You're from: <br/>{locationData.metro_details[0][0]["name"]}
    <br/><br/>
    Most similar: <br/>{locationData.metro_details[1][0]["name"]}

    </p>
  </div>
  )
}
