# London Metering Data
  * The data was composed of two(2) sets of CSV's.  One containing energy values on every half hour and one containing weather values on every hour.  


## Cleaning and joining the data
  *  The two set of data were joined on the time stamp associated with them.  Half hour data was dropped due to convenience.  
  * The metering data came in a strings where there were there should have been floats.  This data needed to be coverted to numeric values.
  * Data had to be one hot encoded for some of the weather features.

  ![](images/Scatter_matrix_of_MAC000002.png)

## Initial Model
  * Initially I a LinearRegression, RidgeCV, and LassoCV on one block of data.  This included the one hot encoded values of the meters. The scores where not great.

![](images/Initial_reg_scores.png)

![](images/Initial_model_feats.png)


# Making a Better Model
  * Removing and focusing on a single meter inside the block was the next task.
  * I used lasso regression and had bad scores >0.03.  I did 2 removals of features ans still had issues with the model not fitting. 
