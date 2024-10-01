

from get_variables import get_census_variables

kmeans_vars = [
    'DP04_0026PE',
    'DP04_0134E',
    'DP04_0026PE',
    'DP04_0026PE',
    'DP04_0134E',
    'DP04_0103PE',
    'DP04_0026PE',
    'DP03_0108PE',
    'DP04_0131E',
    'DP04_0132E'
]


birch_vars = [
    'DP03_0136PE',
    'DP04_0026PE',
    'DP03_0108PE',
    'DP04_0088E',
    'DP04_0134E',
    'DP03_0119PE',
    'DP03_0097PE',
    'DP03_0119PE',
    'DP04_0026PE',
    'DP03_0097PE'
]


variables = get_census_variables([2023])

kmeans_vars = [variables[var] for var in kmeans_vars if var in variables]
"""
[
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
    'Percent!!SELECTED MONTHLY OWNER COSTS (SMOC)!!Housing units without a mortgage!!Less than $250',
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population 19 to 64 years!!In labor force:!!Employed:!!No health insurance coverage',
    'Estimate!!GROSS RENT!!Occupied units paying rent!!$2,000 to $2,499',
    'Estimate!!GROSS RENT!!Occupied units paying rent!!$2,500 to $2,999'
]
"""
birch_vars = [variables[var] for var in birch_vars if var in variables]

"""[
    'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All people!!People in families',
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population 19 to 64 years!!In labor force:!!Employed:!!No health insurance coverage',
    'Estimate!!VALUE!!Owner-occupied units!!$1,000,000 or more',
    'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)',
    'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All families',
    'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!With health insurance coverage!!With private health insurance',
    'Percent!!PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL!!All families',
    'Percent!!YEAR STRUCTURE BUILT!!Total housing units!!Built 1939 or earlier',
    'Percent!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!With health insurance coverage!!With private health insurance'
]
"""

print(kmeans_vars)
print(birch_vars)


"""printed pretty


"""

