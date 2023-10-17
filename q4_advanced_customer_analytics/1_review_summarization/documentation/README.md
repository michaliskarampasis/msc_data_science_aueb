## REVIEW SUMMARIZATION 

### BELOW ARE THE STEPS TO RUN THE PROJECT:

#### 1st STEP:

**reviews_env.yaml**

In this step, it is advisable to activate the environment to run the project.

1. Run the command: `conda env create -f Desktop\reviews_env.yaml`
2. Activate the environment named 'nike_reviews'.

#### 2nd STEP:

**scrape.ipynb**

In this step, we navigate to Nike's website, choose a product, and download all related reviews in a CSV file.

1. Open the file `scrape.ipynb` and run the notebook.
2. A pop-up message will prompt you to select a product.
3. After product selection, execute the `scrape()` function to access Nike's website, search for the product, and download all reviews into a CSV file.

#### 3rd STEP:

**summary.ipynb**

In this step, we locate the path of the CSV file, import it, and analyze the reviews to produce a summary for the selected product.

1. Open the file `summary.ipynb`.
2. Run the function named `summarize()`.
3. Upon completion, the steps executed by the function will be displayed on your screen, and a PDF file containing a summary of all reviews for the selected product will have been created.

