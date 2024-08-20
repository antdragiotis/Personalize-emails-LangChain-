#Personalized emails - Labeling with Named Entity Recognition (NER)
#Use of a structure of defined features and LLM to extract data from the Amazon Fine Food Reviews (the most 1000 reviews are used) The definitions of these features are used in the below Classification class

import pandas as pd
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Parameters 
GPT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

SourceFileName = r'./source_data/Amazon Fine Food Reviews_1000 Recent.csv'
LabeledFileName = r'./intermediate_data/Amazon_Reviews_Labeled.csv'

tqdm.pandas()

def Read_Source_File():
    # Reading the source file : Amazon Fine Food Reviews (the most 1000 reviews) 
    
    source_df = pd.read_csv(SourceFileName)

    # Transfering extracted data to new DataFrame (destimation) with only the necessary fields
    dest_df = pd.DataFrame({
        'ProductId': source_df['ProductId'],
        'UserId': source_df['UserId'],
        'Score': source_df['Score'],
        'Comment': source_df['Summary'] +";" + source_df['Text']
    })
    return dest_df

def Initialize_LLM():
    
    # Definiing the prompt for extracting the structured data
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    Please extract the specified properties from the following comments related to products purchased online, 
    as outlined in the 'Classification' function.
    When a comment mentions two different products with contrasting sentiments, ensure that the correct properties 
    are associated with each product.
    Please give  priority to negative phrases in the text for you to come to a conclusion.
    If you are unable to determine a specific property from the comment, indicate a low confidence level 
    in the 'confidence' property.
    Only extract the properties mentioned in the 'Classification' function.
    Passage:
    {input}
    """
    )
    class Classification(BaseModel):
        product: str = Field(description=
            "The product that the buyer is commenting on.")
        sentiment: str = Field(description=
            "The sentiment expressed in the text, categorized as: 'very negative,' 'negative,' 'neutral,' 'positive,' or 'very positive.")
        aggressiveness: int = Field(description=
            "The level of aggressiveness in the text, rated on a scale from 1 to 10.")
        satisfaction: str = Field(description=
            """The user's satisfaction with the product, rated on a scale from 1 to 5, where 1 indicates dissatisfaction 
            and 5 indicates satisfaction.""")
        confidence: str = Field(description=
            """The confidence level in the extracted properties, rated on a scale from 1 to 5, 
            where 1 indicates low confidence and 5 indicates high confidence.""")

    # LLM initialization
    llm = ChatOpenAI(temperature=TEMPERATURE, model=GPT_MODEL).with_structured_output(Classification)
    labeling_chain = tagging_prompt | llm
    return labeling_chain

def Labels_Assignment(row):

    # Using LLMs response to get the structured data and amend the new DataFrame
    comment = row['Comment']
    response = labeling_chain.invoke({"input": comment})
    row['Product'] = response.product
    row['Sentiment'] = response.sentiment
    row['Aggressiveness'] = int(response.aggressiveness)
    row['Satisfaction'] = int(response.satisfaction)
    row['Confidence'] = int(response.confidence)
    return row

# Reading the source file : Amazon Fine Food Reviews (the most 1000 reviews) 
dest_df = Read_Source_File()

labeling_chain = Initialize_LLM()

# Process to read the source DataFrame and Amend the DataFrame with the Labeling data 
dest_df = dest_df.progress_apply(Labels_Assignment, axis=1)

# Saving the destimation DataFrame as intermediate data to be used by the next processes
dest_df.to_csv(LabeledFileName, index=False)