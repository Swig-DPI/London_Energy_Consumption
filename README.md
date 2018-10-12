# London Metering Data
  * The data was composed of two(2) sets of CSV's.  One containing energy values on every half hour and one containing weather values on every hour.  



## Cleaning and joining the data
  *  The two set of data were joined on the time stamp associated with them.  Half hour data was dropped due to convenience.  
  * The metering data came in a strings where there were there should have been floats.  This data needed to be coverted to numeric values.
  * Data had to be one hot encoded for some of the weather features.

  ![](images/Scatter_matrix_of_MAC000002.png)

  ## Initial Data analysis
    * Energy values vary widely between meters and time.


  ![](images/Energy_all_meters_by_hr.png)
  ![](images/Energy_6_meters_by_hr.png)
  ![](images/Energy_single_meter_by_hr.png)
  ![](images/AVG_energy_for_all_blocks.png)

## Initial Model
  * Initially I used a LinearRegression, RidgeCV, and LassoCV on one block of data without time values.  This included the one hot encoded values of the meters. The scores where not great.

##### Block Scores for all meters
![](images/Initial_reg_scores.png)

![](images/Initial_model_feats.png)


##### Meter Scores for single meter

![](images/Single_meter_inital_reg.png)

  * With this model I began removing features that had a zero (0) in the lasso coefficients.  This made the models only slightly better

##### Single Meter with LassoCV Reduction
![](images/Single_meter_lasso_cv_reg.png)


# Making a Better Model

  * I added in hour of day and day of week into model.  I though this would make a significant impact.  It surprisingly did not. It thought that the time of day and day of week would play a larger role in the energy consumption prediction.  

  * The prediction scores went up by about 0.1 on the block model and by 0.2 on the single meter model.  This is not as high as I expected.  


  *  With this I again began to remove  features that had a zero (0) in the lasso coefficients.  This improved the model only slightly again.

  ![](images/Lasso_reg_with_time.png)

  *  Could  it be just the meter is hard to model?
    * Nope the average and median where:  0.24


  * Did the Block fair any better with the added time parameter?
    * Yes it did
