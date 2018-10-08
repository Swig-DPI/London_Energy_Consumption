# London Energy and Weather
###### Scott Wigle

## Goal
* Predict energy use based on weather and block in the London area.  I will do an initial analysis on a few of the metrics to get averages and see if there is correlation between metrics.    

## Data
* The data in the set is electrical metering data from blocks located in London. Each block is subdivided by the meters located on the block.  The energy data has both daily and half hour incremented data for the years 2011 to 2014. This energy data has the the mean, median, min, max, and standard deviation, see Figure 1. The weather data is daily weather reports from 2011-2014 with many parameters, see Figure 2.  These two sets will be merged together for analysis and building of regression model.

![Meter](/home/smw/Documents/galvanize/capstone_zone/capstone_1/images/Lon_meter_data.png)
Figure 1: Daily energy data for London block 1.

![Data](/home/smw/Documents/galvanize/capstone_zone/capstone_1/images/Lon_weather.png)
Figure 2: Daily weather for London.

## MVP
* Create a linear regression using techniques from class to predict average energy.  I want to know if it is possible to build a good model to predict the energy based off weather information. I will also be curious to see if some blocks use more energy than others. For MVP I will limit the model to only a few blocks depending on compute time.  I will use all blocks if it is not to computationally heavy.

### MVP +
* Use a feature ranking tool such as sklearn.feature_selection.RFE() to fid the best overall parameters to use in my model.  
* Make prediction based off the meters inside each of the blocks.  To do this it would sectionize out the blocks into smaller sections and increase complexity of coding and model.
* Map each block to type of block, residential, industrial, commercial.
* See if daily swings in energy use are similar on all days.


### MVP ++
* Add prediction based off previous days energy data from other blocks and itself.
* Add housing data like income, age, kids in home and other fun factors that could contribute.
* I would also like to add in the estimated cost to the meter user based off the predications.
