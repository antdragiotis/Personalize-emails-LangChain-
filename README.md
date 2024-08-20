## Personalized Emails Generation (OpenAI, LangChain)
This application leverages OpenAI and LangChain functionalities to extract structured data from unstructured text, specifically customer comments on received products. The processed data is then used to cluster comments into distinct segments, enabling the creation of personalized email campaigns. These targeted emails are designed to promote a new product, enhancing marketing effectiveness and customer engagement.

The current application is not to be used for direct integration into a production environment. Instead, it serves as a demonstration of how OpenAI models can be leveraged to customize communications based on historical events and data.

### Purpose 
In today's competitive market, effective communication between businesses and their customers is essential, especially when introducing new products or services. Tailoring these messages to reflect the unique relationship a customer has with a business can significantly enhance engagement. 
This project explores a sequence of steps that leverage customer comments to classify their acceptance and satisfaction levels with the company's products. By analyzing these insights, businesses can generate personalized promotional content when launching new products. This approach improves communication by aligning it more closely with customers' past experiences and perceptions, thereby fostering a more meaningful connection.

### Process Overview 
The generation of emails contents follows the below steps: 

![Process Overview](Personalized_Emails_Overview.png)

### Features
- **Source Data**: The application uses as source data file the **Amazon Fine Food Reviews** (https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
This dataset comprises reviews of fine foods sourced from Amazon. The most recent 1,000 records have been selected for this analysis. The fields labeled *Summary* and *Text* have been combined into a new field titled *Comment*, which is to be utilized in the labeling process by the LLM.
- **Labeling process**: The execution of *PersonalizedEmails_LABELING.py* invokes OpenAI LLM model to extract from each *Comment* a required data structure:
  - **product**: as the product that the buyer is commenting on
  - **sentiment**: as the sentiment expressed in the text, categorized as: 'very negative,' 'negative,' 'neutral,' 'positive,' or 'very positive
  - **aggressiveness**: as the level of aggressiveness in the text, rated on a scale from 1 to 10
  - **satisfaction**: as the user's satisfaction with the product, rated on a scale from 1 to 5
  - **confidence**: as the confidence level in the extracted properties, rated on a scale from 1 to 5. The confidence level is used to filter the data: records with low confidence are excluded from the next steps of the process (Clustering).  
The process saves the results as *intermediate_data/Amazon_Reviews_Labeled.csv*
- **Clustering**: The execution of the *PersonalizedEmails_CLUSTERING.py* excludes records with customers' feedback discrepancies (i.e. when the Score given by a customer is quite different from the *sentiment* of *satisfaction* level addressed by the LLM). The *kMeans* model is used to cluster the data and the results are saved in the *intermediate_data/Amazon_Reviews_Clustered.csv* file.  The Jupyter file *PersonalizedEmails_CLUSTERING.ipynb* is also available as it contains useful visualization outputs. 
- **Text Generation**: The execution of the *PersonalizedEmails_Text_Generation.py* script processes the output from the clustering process by mapping the numerical values of each cluster, specifically the mean values, to corresponding categorical values. These categorical values are then utilized to create a prompt for each cluster, which is subsequently used to generate email content with the use of the LLM. The generated email content is then saved in the file *results/AmazonReviews_WithEmails.csv*. In the *results* directory there is also the file *Sample of generated Personalized Emails.txt* with the result of the Text Generation process. 

- ### How to run the app:
- https://github.com/antdragiotis/Personalize-emails-LangChain-
- change directory to cloned repository cd your-folder-path
- pip install -r requirements.txt 
- run the three processes described above as: 
  - PersonalizedEmails_LABELING.py
  - PersonalizedEmails_CLUSTERING.py
  - PersonalizedEmails_Text_Generation.py

You get the results in the *results/AmazonReviews_WithEmais.csv* file

### Project Structure
- *.py: main application code
- source_data: directory with the source data file
- intermediate_data: directory with intermediate data files that facilitate control and validations
- results: directory with the final output file
- README.md: project documentation