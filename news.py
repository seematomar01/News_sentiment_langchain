import streamlit as st
import pandas as pd
import requests
import re
import requests
import config as cfg
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def get_xausd_news(ticker):
    url = "https://stocknewsapi.com/api/v1"
    params = {
        "tickers": ticker,
        "items": 5,
        "token": "6cnrpqqybevytsorawvqhsvgtweelaxmve9ocbe2"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data:
            data_dataframe = pd.DataFrame(data['data'])
            df = data_dataframe[['date', 'text', 'sentiment', 'source_name']]
            return df
        else:
            return "No news available for the provided ticker."
    else:
        return None
        
#data cleaning        
def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()
    
    
#Now we will use Langchain to form an LLM chain with the OpenAI model.   
llm = ChatOpenAI(model = "gpt-3.5-turbo",
             openai_api_key = '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', 
             temperature = 0)   
             
##template = """
#Identify the sentiment towards the  stocks from the news article , where the sentiment score should be from -10 to +10 where -10 being #the most negative and +10 being the most positve , and 0 being neutral
#give me sentiment and Also give the proper explanation for your answers and how would it effect the prices of different stocks
#Article : {statement}
#"""

template = """
You are an advanced Financial Advisor ,you need to analyse all the news carefully ,news are seprated by delimeter pipe ,You need to give Buy ,Sell,Hold Suggestion ,and also give proper explaination why you suggesting.and Also suggest what is the Right time to Buy then
Collected news : {statement}
"""      
             
def main():
    st.title('Stock News Viewer')

    ticker = st.text_input('Enter Ticker Symbol (e.g., AAPL, MSFT):')
    if st.button('Get News'):
        if ticker:
            news_data = get_xausd_news(ticker)
            prompt = PromptTemplate(template = template, input_variables = ["statement"])
            llm_chain = LLMChain(prompt = prompt, llm = llm)
            sents = []
            ss = ''
            for i in news_data['text'].to_list():
                ss+=i + "|"
            print(ss)    
            sents= llm_chain.run(ss) 
            # news_data['openAI_sentiment']= pd.Series(sents)    
            if isinstance(news_data, pd.DataFrame):
                st.write(news_data)
            else:
                st.write(news_data)
                st.write('Please check your ticker symbol and try again.')

        st.write(sents)
        
if __name__ == "__main__":
    main()



###################################################final code#####################################################################

def get_xausd_news(ticker):
    url = "https://stocknewsapi.com/api/v1"
    params = {
        "tickers": ticker,
        "items": 5,
        "token": "6cnrpqqybevytsorawvqhsvgtweelaxmve9ocbe2"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data:
            data_dataframe = pd.DataFrame(data['data'])
            df = data_dataframe[['date', 'text', 'sentiment', 'source_name']]
            llm = ChatOpenAI(model = "gpt-3.5-turbo",
            openai_api_key = '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', 
            temperature = 0)   
            template = """
You are an advanced Financial Advisor ,you need to analyse all the news carefully ,news are seprated by delimeter pipe ,You need to give Buy ,Sell,Hold Suggestion ,and also give proper explaination why you suggesting.and Also suggest what is the Right time to Buy then
Collected news : {statement}
    """     prompt = PromptTemplate(template = template, input_variables = ["statement"])
            llm_chain = LLMChain(prompt = prompt, llm = llm)
            sents = []
            ss = ''
            for i in df['text'].to_list():
                ss+=i + "|"
            print(ss)    
            sents= llm_chain.run(ss)       
            return sents
        else:
            return "No news available for the provided ticker."
    else:
        return None
        