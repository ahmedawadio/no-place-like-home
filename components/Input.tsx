"use client";

import React, { useState } from "react";
import { PlaceholdersAndVanishInput } from "./ui/placeholders-and-vanish-input";

interface InputProps {
  setIsLoading: (loading: boolean) => void; // Function to set loading state
  setIsSubmitted: (submitted: boolean) => void; // Function to set submission state
  setSubmittedInput: (input: string) => void; // Function to set the submitted input
}

export function Input({ setSubmittedInput, setIsLoading, setIsSubmitted }: InputProps) {
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

  // State variables for error handling
  const [error, setError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [shouldAnimate, setShouldAnimate] = useState(false); // New state to trigger animation

  // Handle input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Reset error if input becomes exactly 5 digits
    if (error && /^\d{5}$/.test(e.target.value)) {
      setError(false);
      setErrorMessage("");
    }
  };

  // Handle form submission
  const onSubmit = (inputValue: string) => {
    setIsLoading(true);
    setIsSubmitted(true);

    const trimmedValue = inputValue.trim();

    // Validation: Check if input is exactly 5 digits
    if (!/^\d{5}$/.test(trimmedValue)) {
      setError(true);
      setErrorMessage("Zipcode must contain exactly 5 digits.");
      setIsLoading(false);
      setIsSubmitted(false);
      setShouldAnimate(false);
      return; // Prevent submission
    }

    // If validation passes
    setError(false);
    setErrorMessage("");
    setSubmittedInput(trimmedValue);
    setShouldAnimate(true); // Trigger animation
  };

  return (
    <div className="my-2 w-full flex flex-col items-center justify-center">
      {/* Conditionally render error message */}
      {error && (
      <p className="absolute top-3 text-red-500 text-sm  text-center" id="zipcode-error">
          {errorMessage}
        </p>
      )}
      {/* Pass the necessary props */}
      <PlaceholdersAndVanishInput
        placeholders={placeholders}
        onChange={handleChange}
        onSubmit={onSubmit}
        error={error}
        shouldAnimate={shouldAnimate}
        setShouldAnimate={setShouldAnimate}
      />
    </div>
  );
}
