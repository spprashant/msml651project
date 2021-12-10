# MSML 651 Project
## NYC Taxi Fare Prediction


## Code.Files
The actual code for this project is available in two formats:  
* IPYNB - MSML651Project.ipynb  
* PY - MSML651Project.py  
  
This code was original developed on the Databricks platform which has its own DBFS filesystem.  
Cells which deal with reading and writing from the filesystem will need to be changed to the  
appropriate filesystem before attempting to run the files.
  
  
## Cross Validation. Output
The MLFlow.output of the cross validation step are available in the repo  
* Linear Regression - linear_regression_cv.csv
* Random Forest - random_forest_cv.csv
* Gradient Boosted Trees - grad_boost_cv.csv

## Test File Output
The output of the final test file runs are available in this box location  
[BOX](https://umd.box.com/s/st68nr6l622gea58cwknel0vf6w3fytx)  
  
There are three zips in that location, each for one model:  
* Linear Regression - lr_test_output.zip  
* Random Forest - rf_test_output.zip  
* Gradient Boosted Trees - gb_test_output.zip  

The files are split into two, since Databricks has some issues downloading large files.