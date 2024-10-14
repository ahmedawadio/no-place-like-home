"use client";
import Image from "next/image";
import React from "react";
import { Carousel, Card } from "@/components/ui/apple-cards-carousel";
import { Home, Users, Briefcase, GraduationCap, HomeIcon, Heart, MapPin } from "lucide-react";
import { MetroDetail } from "@/types";


// Define the type for individual card items
interface CardData {
    category: string;
    title: string;
    src: string;
    content: React.ReactNode;
  }

type Card = {
    src: string;
    title: string;
    category: string;
    content: React.ReactNode;
    data_source: string;
  };
  


  interface Props {
    metro_details: MetroDetail[];
  }

export function MetroCarousel({metro_details}: Props) {


    interface MetroContentProps {
        metro: MetroDetail;
      }
      
      const MetroContent: React.FC<MetroContentProps> = ({ metro }) => {
        // Function to split content by \n\n and wrap each in a <p> tag
        const renderParagraphs = (text: string) => {
          return text.split('\n\n').map((paragraph, index) => (
            <p key={index} className="mb-4">
              {paragraph}
            </p>
          ));
        };
      
        // Check if all sections are empty
        const isAllSectionsEmpty = 
          !metro.about && 
          !metro.population && 
          !metro.economy && 
          !metro.education && 
          !metro.housing && 
          !metro.health;
      
        return (
          <>
            {isAllSectionsEmpty ? (
              <div className="text-neutral-400 text-center text-2xl font-sans p-8">
                No demographic details provided.
              </div>
            ) : (
              <>
                {/* About Section */}
                {metro.about && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <MapPin className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">About</span>
                      <br /><br />
                      {renderParagraphs(metro.about)}
                    </div>
                  </div>
                )}
                
                {/* Population & Diversity Section */}
                {metro.population && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <Users className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">Population & Diversity</span>
                      <br /><br />
                      {renderParagraphs(metro.population)}
                    </div>
                  </div>
                )}
                
                {/* Economy Section */}
                {metro.economy && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <Briefcase className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">Economy</span>
                      <br /><br />
                      {renderParagraphs(metro.economy)}
                    </div>
                  </div>
                )}
                
                {/* Education Section */}
                {metro.education && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <GraduationCap className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">Education</span>
                      <br /><br />
                      {renderParagraphs(metro.education)}
                    </div>
                  </div>
                )}
                
                {/* Housing & Living Section */}
                {metro.housing && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <HomeIcon className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">Housing & Living</span>
                      <br /><br />
                      {renderParagraphs(metro.housing)}
                    </div>
                  </div>
                )}
                
                {/* Health Section */}
                {metro.health && (
                  <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start justify-center">
                    <Heart className="text-neutral-400 w-8 h-8 mr-4 flex-shrink-0" />
                    <div className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl">
                      <span className="font-bold text-neutral-200">Health</span>
                      <br /><br />
                      {renderParagraphs(metro.health)}
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        );
      };
      
      

      
      
//   const cards = data.map((card, index) => (
//     <Card key={card.src} card={card} index={index} />
//   ));

  const cards = metro_details.map((metro, index) => {
    const cardData = {
      category: `Metro Area ${index + 1}`,
      title: metro.name,
      src: metro.image_uri,
      content: <MetroContent metro={metro} />,
      data_source: metro.data_source,
    };

    return (
      <Card key={metro.mid} card={cardData} index={index} />
    );
  });
  return (
    <div className="w-full h-full py-0 overflow-x-visible">
 
    <div className="overflow-x-visible">
      <Carousel items={cards} />
    </div>
  </div>

  );
}



// const DummyContent = () => {
//   return (
//     <>
//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <MapPin className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">About</span> <br /><br />
//           In 2022, Big Stone Gap, VA had a population of 39.8k people with a median age of 42 and a median household income of $46,680...
//         </p>
//       </div>

//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <Users className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">Population & Diversity</span> <br /><br />
//           Big Stone Gap, VA is home to a population of 39.8k people, from which 99.3% are citizens. As of 2022, 1.62% of Big Stone Gap, VA residents were born outside of the country...
//         </p>
//       </div>

//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <Briefcase className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">Economy</span> <br /><br />
//           The economy of Big Stone Gap, VA employs 14.2k people. The largest industries in Big Stone Gap, VA are Health Care & Social Assistance, Retail Trade, and Educational Services...
//         </p>
//       </div>

//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <GraduationCap className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">Education</span> <br /><br />
//           In 2022, universities in Big Stone Gap, VA awarded 1,208 degrees. The student population is skewed towards women, with 1,404 male students and 2,228 female students...
//         </p>
//       </div>

//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <HomeIcon className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">Housing & Living</span> <br /><br />
//           The median property value in Big Stone Gap, VA was $107,600 in 2022, reflecting a 13.3% increase from $95,000 in the previous year. The homeownership rate is 68.6%...
//         </p>
//       </div>

//       <div className="bg-neutral-800 p-8 md:p-14 rounded-3xl mb-4 flex items-start">
//         <Heart className="text-neutral-400 w-20 pr-4 mt-1" />
//         <p className="text-neutral-400 text-base md:text-2xl font-sans max-w-3xl mx-auto">
//           <span className="font-bold text-neutral-200">Health</span> <br /><br />
//           In 2022, 91.9% of the population of Big Stone Gap, VA had health coverage. Primary care physicians in Virginia see 1,324 patients per year on average...
//         </p>
//       </div>
//     </>
//   );
// };

  

//   const data = [
//     {
//       category: "Number 1",
//       title: "Worcester, MA-CT Metro Area",
//       src: "/assets/Downtown Albany-Schenectady-Troy.webp",
//       content: <DummyContent />,
//       data_source: "https://datausa.io/profile/geo/big-stone-gap-va-31000US13720",
//     },
//     {
//       category: "Number 2",
//       title: "Springfield, MA Metro Area",
//       src: "/assets/Downtown Springfield MA.webp",
//       content: <DummyContent />,
//       data_source: "https://datausa.io/profile/geo/big-stone-gap-va-31000US13720",
//     },
//     {
//       category: "Number 3",
//       title: "Rochester, NY Metro Area",
//       src: "/assets/Downtown Rochester Pastel.webp",
//       content: <DummyContent />,
//       data_source: "https://datausa.io/profile/geo/big-stone-gap-va-31000US13720",
//     },
//     {
//       category: "Number 4",
//       title: "Syracuse, NY Metro Area",
//       src: "/assets/Downtown Syracuse Pastel Scene.webp",
//       content: <DummyContent />,
//       data_source: "https://datausa.io/profile/geo/big-stone-gap-va-31000US13720",
//     },
//     {
//       category: "Number 5",
//       title: "Hartford-East Hartford-Middletown, CT Metro Area",
//       src: "/assets/Downtown Hartford Scene.webp",
//       content: <DummyContent />,
//       data_source: "https://datausa.io/profile/geo/big-stone-gap-va-31000US13720",
//     },
//   ];
  
