"use client";

import React, { useState, useEffect } from "react";
import { ChevronsUpDown, ChevronDown, TrendingUpDown } from "lucide-react";

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
import { Variable } from "@/types";

interface VariableComboboxPopoverProps {
  variables: Variable[];
  selectedVariable: Variable | null;
  setSelectedVariable: React.Dispatch<React.SetStateAction<Variable>>;
}

export function VariableComboboxPopover({
  variables,
  selectedVariable,
  setSelectedVariable,
}: VariableComboboxPopoverProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!selectedVariable && variables.length > 0) {
      setSelectedVariable(variables[0]);
    }
  }, [variables, selectedVariable, setSelectedVariable]);

  return (
    <div className="flex bg-card items-start space-x-2 justify-start flex-col">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>

          <div
            className="border  px-4 border-gray-800 rounded-3xl py-3 w-full cursor-pointer hover:bg-gray-900"
            onClick={() => setOpen(true)}
          >
            <h3 className="text-xs font-semibold tracking-tight flex items-center">
          <TrendingUpDown className="w-4 h-4 mr-2" />
              <span>{selectedVariable ? selectedVariable.name : "Select Variable"}</span>
              <ChevronDown className="w-4 h-4 ml-6" />
            </h3>
          </div>
        </PopoverTrigger>
        <PopoverContent className="p-0 w-[400px]" side="bottom" align="start">
          <Command>
            <CommandInput placeholder="Change variable..." />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup>
                {variables.map((variable, index) => (
                  <CommandItem
                    key={variable.variable_code}
                    value={variable.name}
                    onSelect={() => {
                      setSelectedVariable(variable);
                      setOpen(false);
                    }}
                    className={cn(
                      "cursor-pointer px-4 py-2", // Default styles for list items
                      selectedVariable?.variable_code === variable.variable_code && "bg-gray-700" // Highlight selected item
                    )}
                  >
                    <span className="flex items-center space-x-2">

                      <span>{variable.name}</span>
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
