# MSML 651 Project
## NYC Taxi Fare Prediction


## Code Files
The actual code for this project is available in two formats:  
* IPython Notebook - [MSML651Project.ipynb](MSML651Project.ipynb)  
* Python File - [MSML651Project.py](MSML651Project.py)  
  
This code was original developed on the Databricks platform which has its own DBFS filesystem.  
Cells which deal with reading and writing from the filesystem will need to be changed to the  
appropriate filesystem before attempting to run the files.
  
  
## Original Dataset
The original dataset can be downloaded from this [Kaggle repo](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)  
Kaggle might ask for an account and signing certain T&C if you need access to the files  
  
  
## Cross Validation Output
The MLFlow output of the cross validation step are available in the repo in the `cv` folder  
* Linear Regression - [linear_regression_cv.csv](cv/linear_regression_cv.csv)
* Random Forest - [random_forest_cv.csv](cv/random_forest_cv.csv)
* Gradient Boosted Trees - [grad_boost_cv.csv](cv/grad_boost_cv.csv)
  
  
## Test File Output
The output of the final test file runs are available in this box location  
[BOX](https://umd.box.com/s/st68nr6l622gea58cwknel0vf6w3fytx)  
  
There are three zips in that location, each for one model:  
* Linear Regression - lr_test_output.zip  
* Random Forest - rf_test_output.zip  
* Gradient Boosted Trees - gb_test_output.zip  

The files are split into two, since Databricks has some issues downloading large files.
The columns of interest are the `fare_amount` (actual value) and `prediction` (predicted_value).  
  
  
## Final RMSE Values
| Model                  | Final RMSE |
| ---------------------- | ---------- |
| Linear Regression      | 5.31       |
| Random Forest          | 4.31       |
| Gradient Boosted Trees | 4.08       |
