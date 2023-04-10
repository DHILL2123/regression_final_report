# regression_final_report

# Project Description 
We have been tasked with reviewing zillow property data for 2017. The data consists of home features like the number of bedrooms and bathrooms. The area of the home and its location to name a few features. All these things go into assessing the tax value of a property.

# Project Goal 
* The goal of this project is to create a model that predicts the tax value of a single family property better than the current baseline model using 2017 property data. 
* We will do this without using any of the financial features available for the property. 

# Initial Thoughts
## Key features in deciding tax value
* Area 
* Bedroom count
* Bathroom count
* Location 
* Building structure

# Plan of action

* Acquire the data from MySQL Workbench

* Prepare data

* Update columns from existing data
    
* Explore data to find correlations for tax value 

* Answer the following initial questions
    * Does area of a home affect tax value
    * Does the bedroom count affect tax value
    * Does the bathroom count affect tax value
    * Does the number of stories affect tax value
    * Does the material the structure is made of affect tax value
    
* Build a model based on key features that correlate with tax value

* Evaluate models on train and validate data

* Select the best model based on lowest error score

* Evaluate the best model on test data

* Draw conclusions

# Steps to reproduce
* Clone this repo.
* Acquire the data from MYSQL Workbench
* Put the data in the file containing the cloned repo.
* Run notebook.


# Takeaways
* Our Polynomial Model performed best out of the four created models.
* Baseline is $195,345.66
* Model is $177,884.32
* Our model beat baseline by $17,461.34

# Conclusion
* Exploration proved the correlation of our features so accounting for area, number of bathroomss, and bedrooms will help better predict a homes tax value.
*Our model out performs baseline and would bring our predictions $17,461 closer to the actual value of the home.

# Recommendations
* Implement our model, we can use this to our advantage when selling marketing to real estate agents and investing groups



