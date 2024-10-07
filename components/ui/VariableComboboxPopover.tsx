"use client";

import React, { useState, useEffect } from "react";
import { ChevronsUpDown } from "lucide-react";

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
    <div className="flex items-start space-x-2 justify-start flex-col">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <div
            className="border border-gray-800 rounded-lg p-4 w-full cursor-pointer hover:bg-gray-900"
            onClick={() => setOpen(true)}
          >
            <h3 className="text-3xl font-semibold tracking-tight flex items-center space-x-2">
              <span>{selectedVariable ? selectedVariable.name : "Select Variable"}</span>
              <ChevronsUpDown className="w-5 h-5" />
            </h3>

            <p className="mt-2 opacity-50 flex items-center space-x-2">
              <span>select a variable</span>
            </p>
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
                      "cursor-pointer px-4 py-2 hover:bg-red-800", // Default styles for list items
                      selectedVariable?.variable_code === variable.variable_code && "bg-gray-700" // Highlight selected item
                    )}
                  >
                    <span className="flex items-center space-x-2">
                      <span className="flex items-center justify-center w-6 h-6 border border-gray-500 rounded-full">
                        {index + 1}
                      </span>
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
