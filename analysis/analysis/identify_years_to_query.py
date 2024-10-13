"""
My goal is to use a model that has a large amount of variables used to describe a locaiton,
in a resonable amount of query time. Because of this, I will use the  Census api on
the ACS 1 and 5 years. I am choosing the Profile api, because the larger data tables are mostly subsets The dataset has the most popular and relevant averages for each 
geographic location. 

In order to choose which variables ot query for each year, I will see which variables are being
contiuousy used across the years. 

The more time we go back the better, but the worse the overlapping variables. 

I will choose the year right before the big drop. I chosse 2019-2023(most recent year).

"""

import matplotlib.pyplot as plt
from get_variables import get_census_variables

x=[]
y=[]



for year in range(2015, 2023+1):
    print(year)
    x.append(year)
    y.append(len(get_census_variables(list(range(year, 2023+1)))) )


# Plot the line chart
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Number of Variables')

# Add titles and labels
plt.title('Census Variable Overlaps Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Variables')

# Show the grid
plt.grid(True)

# Display the chart
plt.legend()
plt.show()
