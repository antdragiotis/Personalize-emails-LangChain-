# Personalized Emails - Text Generation
#This file gets the results of the Clustering process and maps the numerical values of each Cluster (actually the mean values) to categorial values. These #categorial values are used to formulate a prompt per Cluster and ask a LLM to generate email context. 
#In this way the generated email content is in accordance with the feature values of each cluster. 
#This outcome may be used by a process that combines a customers file, with the email text per Cluster to send 'personalized' messages to customers. 
#There are few cases of multiple comments records for the same customer but in such cases it is found that the customer's acceptance of the products is the same #for these multiple comments records and therefore the customer is assigned to the same Cluster. 
#Just to note that the purpose of this project is to show a use case of LangChain / LLM functionality used in business tasks and not to provide software that can #run in production.

import pandas as pd
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Parameters 
GPT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 250
TEMPERATURE = 0.5

llm = ChatOpenAI(model=GPT_MODEL, temperature = TEMPERATURE, max_tokens = MAX_TOKENS)

pd.options.display.float_format = '{:,.2f}'.format 
tqdm.pandas()

CusteringResultsFileName = r'./intermediate_data/Amazon_Reviews_ClusteringResults.csv'
ReviewsWithEmaisFileName = r'./results/AmazonReviews_WithEmais.csv'

def Loading_Clustering_Results():
    """ Loading & Transforming Data"""
    
    # LOADING Clustering Results
    df = pd.read_csv(CusteringResultsFileName)

    # Selecting only necessary features 
    clust_df = df[["Cluster","Sentiment_mean","Aggressiveness_mean","Satisfaction_mean","Score_mean"]]
    clust_df = clust_df.map(lambda x: int(round(x, 0)))

    # Mapping of numerical values to categorial values for making these descriptions part of the prompt for the LLM
    # Also for easier validation of the outputs by humans  
    clust_df["Sentiment"] = clust_df["Sentiment_mean"].map({1 : 'very negative', 2 : 'negative', 3 : 'neutral', 
            4 :'positive', 5: 'very positive'})
    clust_df["Aggressiveness"] = clust_df['Aggressiveness_mean'].map({
            1 :'non-aggressive', 2 :'non-aggressive', 
            3: 'mildly assertive', 4: 'mildly assertive', 
            5: 'assertive', 6: 'assertive',
            7: 'firmly critical', 8: 'firmly critical',
            9 : 'stronly aggresive', 10 : 'stronly aggresive'})
    clust_df["Satisfaction"] = clust_df["Satisfaction_mean"].map({1 : 'highly dissatisfied', 2 : 'dissatisfied', 
            3 : 'neutral', 4 :'satisfied', 5: 'fully satisfied'})
    clust_df["Score"] = clust_df["Score_mean"].map({1 : 'very low score', 2 : 'low score', 3 : 'average score', 
            4 :'high score', 5: 'very high score'})
    return clust_df

def Email_Generation(): 
    """ Generation of email contents based on a general prompt and categorial parameters of each Cluster"""
    generation_prompt = ChatPromptTemplate.from_template(
    """
    Please write an email to be sent to customers of a food product company named "NER Products", 
    introducing a new product called "NER Tasty" The email content should be personalized based on the following key features 
    derived from previous customer feedback on products offered by "NER Products".
    "sentiment": The sentiment expressed in the text, categorized as: "very negative", "negative", "neutral", "positive", or "very positive".
    "aggressiveness": The level of aggressiveness in the text.
    "satisfaction": The user's satisfaction with the product.
    "score": The score that customers use to rate the products offered. High scores mean positive customer reactions. 

    Please write the email content  in no more than 130 words, for a pool of customers with the following features values:  
    "sentiment": {sentiment}
    "aggressiveness": {aggressiveness}
    "satisfaction": {satisfaction}
    "score": {score}"
    """   
    )
    generation_chain = generation_prompt | llm

    def Write_email_Content(row):
        """ Generation of email content based on  categorial parameters of the row"""
        response = generation_chain.invoke(
            {"sentiment": row.Sentiment,"aggressiveness": row.Aggressiveness,"satisfaction": row.Satisfaction,"score": row.Score}
        )
        return response.content
    
    clust_df["email_content"] = clust_df.progress_apply(Write_email_Content, axis=1)

    # Saving of Results
    clust_df.to_csv(ReviewsWithEmaisFileName)

       # Showing the Results
    for index, row in clust_df.iterrows():
        print(f"Cluster: {row.Cluster}\n{"-"*120}\nSentiment: {row.Sentiment} | Aggressiveness: {row.Aggressiveness} | Satisfaction: {row.Satisfaction} | Score: {row.Score}\n{"-"*120}\nemail Content:\n{"-"*1}\n{row.email_content}\n\n")

# Initialization of a DataFrame
clust_df = pd.DataFrame()
clust_df = Loading_Clustering_Results()
Email_Generation()