"use client";

import * as React from "react";
import {ChevronDown, ChevronsUpDown, TrendingUpDown,MapPin} from "lucide-react";

import { cn } from "@/lib/utils";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { MetroDetail, MetroMetric, ZipcodeDetails } from "@/types";


interface MetroComboboxPopoverProps {
  metroDetails: MetroDetail[];
  selectedMetro: MetroDetail | null;
  setSelectedMetro: React.Dispatch<React.SetStateAction<MetroDetail >>;
}

export function MetroComboboxPopover({
  metroDetails,
  selectedMetro,
  setSelectedMetro,
}: MetroComboboxPopoverProps) {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    if (!selectedMetro && metroDetails.length > 0) {
      setSelectedMetro(metroDetails[0]);
    }
  }, [metroDetails, selectedMetro, setSelectedMetro]);

  return (
    <div className=" bg-card items-start space-x-2 justify-start flex-col">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>


          <div
            className="border  px-4  border-gray-800 rounded-3xl py-3 w-full cursor-pointer hover:bg-gray-900"
            onClick={() => setOpen(true)}
          >
            <h3 className="text-xs font-semibold tracking-tight flex items-center">
          <MapPin className="w-4 h-4 mr-2" />
              <span>{selectedMetro ? selectedMetro.name : "Select Metro Area"}</span>
              <ChevronDown className="w-4 h-4 ml-6" />
            </h3>

          </div>
        </PopoverTrigger>
        <PopoverContent className="p-0 w-[400px]" side="bottom" align="start">
          <Command>
            <CommandInput placeholder="Change metro area..." />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup>
                {metroDetails.map((metro, index) => (
                  <CommandItem
                    key={metro.mid}
                    value={metro.name}
                    onSelect={() => {
                      setSelectedMetro(metro);
                      setOpen(false);
                    }}
                    className={cn(
                      "cursor-pointer px-4 py-2 hover:bg-red-800", // Default styles for list items
                      selectedMetro?.mid === metro.mid && "bg-gray-700" // Highlight selected item
                    )}
                  >
                    <span className="flex items-center space-x-2">
                      <span className="flex items-center justify-center w-6 h-6 border border-gray-500 rounded-full">
                        {index + 1}
                      </span>
                      <span>{metro.name}</span>
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}