"use client";

import * as React from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { MetroDetail, MetroMetricsGroup, Variable } from "@/types";

interface VariablesGraphProps {
  metroMetrics: MetroMetricsGroup[];
  selectedMetro: MetroDetail | null;
  selectedVariable: string;
  selectedLocations: string[];
  variables: Variable[];
}

export function VariablesGraph({
  metroMetrics,
  selectedMetro,
  selectedVariable,
  selectedLocations,
  variables,
}: VariablesGraphProps) {


  const metroOneMetrics = metroMetrics[0]?.metrics.filter(
    (metric) => metric.variable_code === selectedVariable
  );

  const selectedMetroMetrics = selectedMetro
    ? metroMetrics
        .find((metro) => metro.metro_id === selectedMetro.mid)
        ?.metrics.filter((metric) => metric.variable_code === selectedVariable)
    : [];

  if (
    (!metroOneMetrics || metroOneMetrics.length === 0) &&
    (!selectedMetroMetrics || selectedMetroMetrics.length === 0)
  ) {
    return (
      <div className="text-center text-lg font-semibold text-muted-foreground">
        Can't find variable data, try another..
      </div>
    );
  }
  
  const selectedVariableDetails = variables.find(
    (variable) => variable.variable_code === selectedVariable
  );

  // Memoize chartData to prevent unnecessary recalculations
  const chartData = React.useMemo(() => {
    const allYears = new Set([
      ...(metroOneMetrics?.map((metric) => metric.year) || []),
      ...(selectedMetroMetrics?.map((metric) => metric.year) || []),
    ]);

    const sortedYears = Array.from(allYears).sort();

    return sortedYears.map((year) => {
      const metroOneMetric = metroOneMetrics?.find(
        (metric) => metric.year === year
      );
      const selectedMetroMetric = selectedMetroMetrics?.find(
        (metric) => metric.year === year
      );

      return {
        year: year.toString(),
        metroOneValue: metroOneMetric?.value,
        selectedMetroValue: selectedMetroMetric?.value,
      };
    });
  }, [metroOneMetrics, selectedMetroMetrics]);

  const valueType = selectedVariableDetails?.type;

  const valueFormatter = (value: number) => {
    switch (valueType) {
      case "percent":
        return `${value}%`;
      case "dollars":
        return `$${value.toLocaleString()}`;
      case "count":
      default:
        return value.toLocaleString();
    }
  };

  const chartConfig = {
    metroOneValue: {
      label: `${selectedLocations[0]}`,
      color: "hsl(var(--chart-1))",
    },
    selectedMetroValue: {
      label: `${selectedMetro?.name}`,
      color: "hsl(var(--chart-2))",
    },
  } satisfies ChartConfig;

  // State to track the active data point
  const [activeData, setActiveData] = React.useState(
    chartData[chartData.length - 1]
  );

  // Update activeData only when selectedVariable or selectedMetro change
  React.useEffect(() => {
    setActiveData(chartData[chartData.length - 1]);
  }, [selectedVariable, selectedMetro]);

  return (
    <Card className=" border rounded-2xl  ">
      <CardHeader>
        <CardTitle>{selectedVariableDetails?.name || selectedVariable}</CardTitle>
        <CardDescription>
          {selectedVariableDetails?.description || selectedVariable}
        </CardDescription>
        <div className="flex">
          {[selectedLocations[0], selectedMetro?.name].map((location, index) => {
            const configKey = index === 0 ? "metroOneValue" : "selectedMetroValue";
            const locationColor = chartConfig[configKey].color;

            return (
              <div
                key={location}
                className="mt-6 flex flex-1 flex-col justify-center gap-1 border-t px-6 py-4 text-left even:border-l sm:px-8 sm:py-6"
              >
                <span
                  className="text-xs text-muted-foreground"
                  style={{ color: locationColor }}
                >
                  {location}
                </span>
                <div className="flex items-baseline justify-between">
                <span className="text-lg font-bold leading-none sm:text-3xl">
  {valueFormatter(
    activeData[index === 0 ? "metroOneValue" : "selectedMetroValue"] ?? 0
  )}
</span>

                  <span className="text-xs text-muted-foreground ml-2">
                    {activeData?.year}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </CardHeader>
      <CardContent className="">
      <ChartContainer className="" config={chartConfig}>
          <AreaChart
            data={chartData}
            margin={{
              left: 12,
              right: 12,
            }}
            onMouseMove={(state) => {
                if (state.isTooltipActive && state.activePayload && state.activePayload[0]) {
                  const activePoint = state.activePayload[0].payload;
                  setActiveData(activePoint);
                }
              }}
            onMouseLeave={() => {
              setActiveData(chartData[chartData.length - 1]);
            }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="year"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => valueFormatter(value)}
            />
            <ChartTooltip
              cursor={true}
              content={
                <ChartTooltipContent
                  formatter={(value, name) => {
                    const config = chartConfig[name as keyof typeof chartConfig];
                    return (
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          color: config?.color,
                        }}
                      >
                        <span
                          style={{
                            fontWeight: "bolder",
                            color: "white",
                            marginRight: "8px",
                          }}
                        >
                        {typeof value === "number" ? valueFormatter(value) : value}
                        </span>
                        <span>{config?.label}</span>
                      </div>
                    );
                  }}
                />
              }
            />
            <defs>
              <linearGradient id="fillMetroOne" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-metroOneValue)"
                  stopOpacity={0.8}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-metroOneValue)"
                  stopOpacity={0.1}
                />
              </linearGradient>
              <linearGradient id="fillSelectedMetro" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-selectedMetroValue)"
                  stopOpacity={0.8}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-selectedMetroValue)"
                  stopOpacity={0.1}
                />
              </linearGradient>
            </defs>
            <Area
              dataKey="metroOneValue"
              type="natural"
              fill="url(#fillMetroOne)"
              fillOpacity={0.4}
              stroke="var(--color-metroOneValue)"
            />
            <Area
              dataKey="selectedMetroValue"
              type="natural"
              fill="url(#fillSelectedMetro)"
              fillOpacity={0.4}
              stroke="var(--color-selectedMetroValue)"
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
      <CardFooter>
        <div className="flex w-full items-start gap-2 text-sm">
          <div className="grid gap-2">
            {chartData.length > 0 && (
              <div className="flex items-center gap-2 font-medium leading-none">
                {(() => {
                  const metroOneValue = activeData.metroOneValue || 0;
                  const selectedMetroValue = activeData.selectedMetroValue || 0;

                  // Calculate percentage difference
                  const difference = selectedMetroValue - metroOneValue;
                  const percentageChange =
                    selectedMetroValue !== 0
                      ? parseFloat(
                          ((difference / selectedMetroValue) * 100).toFixed(1)
                        )
                      : 0;
                  const isEqual = difference === 0;
                  const isTrendingUp = difference < 0;

                  return (
                    <>
                      {isEqual ? (
                        <>No change compared to {selectedMetro?.name}</>
                      ) : isTrendingUp ? (
                        <>
                          Trending up by {Math.abs(percentageChange)}% compared to{" "}
                          {selectedMetro?.name} <TrendingUp className="h-4 w-4" />
                        </>
                      ) : (
                        <>
                          Trending down by {Math.abs(percentageChange)}% compared
                          to {selectedMetro?.name}
                          <TrendingDown className="h-4 w-4" />
                        </>
                      )}
                    </>
                  );
                })()}
              </div>
            )}
            {chartData.length > 0 && (
              <div className="flex items-center gap-2 leading-none text-muted-foreground">
                {activeData?.year}
              </div>
            )}
          </div>
        </div>
      </CardFooter>
    </Card>
  );
}

export default VariablesGraph;
