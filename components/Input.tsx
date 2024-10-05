"use client";

import { PlaceholdersAndVanishInput } from "./ui/placeholders-and-vanish-input";


interface InputProps {
  setIsLoading: (loading: boolean) => void; // A function that accepts a boolean
  setIsSubmitted: (submitted: boolean) => void; // A function that accepts a boolean
  setSubmittedInput: (input: string) => void; // A function that accepts a string
}

export function Input({ setSubmittedInput,setIsLoading, setIsSubmitted}: InputProps) {
  const placeholders = [
        "10001 (New York City, NY)",
        "90001 (Los Angeles, CA)",
        "60601 (Chicago, IL)",
        "77001 (Houston, TX)",
        "85001 (Phoenix, AZ)",
        "19101 (Philadelphia, PA)",
        "75201 (Dallas, TX)",
        "94101 (San Francisco, CA)",
        "30301 (Atlanta, GA)",
        "20001 (Washington, DC)",
        "33101 (Miami, FL)",
        "98101 (Seattle, WA)",
        "02101 (Boston, MA)",
        "37201 (Nashville, TN)",
        "48201 (Detroit, MI)",
        "64101 (Kansas City, MO)",
        "73101 (Oklahoma City, OK)",
        "53201 (Milwaukee, WI)",
        "84101 (Salt Lake City, UT)",
        "43201 (Columbus, OH)",
      ];
      
      

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // console.log(e.target.value);
  };

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true)
    setIsSubmitted(true)
    // console.log("submitted", e);
    const target = e.target as HTMLFormElement;
    const inputElement = target.elements[0] as HTMLInputElement; // Assuming your input is the first element in the form
    console.log("submitted", inputElement.value); // Now you can access the value
    setSubmittedInput(inputElement.value)

  };

  return (
    <div className=" my-2 w-full  justify-center  ">

      <PlaceholdersAndVanishInput
        placeholders={placeholders}
        onChange={handleChange}
        onSubmit={onSubmit}
      />
    </div>
  );
}
