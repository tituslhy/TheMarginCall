#%%
import os
from dotenv import load_dotenv, find_dotenv
import warnings
import sys
import time
import re

import streamlit as st

_ = load_dotenv(find_dotenv())
warnings.filterwarnings('ignore')

__curdir__ = os.getcwd()
for folder in ["src", "tools"]:
    sys.path.append(
        os.path.join(
            __curdir__,
            f"./{folder}"
        )
    )

from crewAI.crew_utils import TheResearchCrew
#%%

class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']
        self.color_index = 0 #initialize color index
    
    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        
        #check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()
        
        if task_value:
            st.toast(":robot_face: " + task_value)
        
        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors) #increment color index and wrap around if necessary
            
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain",
                                                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")
        
        if "Principal technical analyst" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Principal technical analyst", f":{self.colors[self.color_index]}[Principal technical analyst]"
            )
        if "Principal fundamental analyst" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Principal fundamental analyst", f":{self.colors[self.color_index]}[Principal fundamental analyst]"
            )
        if "Principal researcher" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Principal researcher", f":{self.colors[self.color_index]}[Principal researcher]"
            )
        if "Principal Finance Reporter" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Principal Finance Reporter", f":{self.colors[self.color_index]}[Principal Finance Reporter]"
            )
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(
                ''.join(self.buffer),
                unsafe_allow_html = True
            )
            self.buffer = []

task_values = []

st.title("The Margin Call")
with st.expander("About the team:"):
    st.subheader("Diagram")
    left, cent, right = st.columns(3)
    with cent:
        st.image("graph.jpeg")
    
    st.subheader("Principal Technical Analyst")
    st.text(
        """
        Role =  Principal technical analyst
        Goal =  Impress everyone with your technical analysis of financial
                market data and strategic investment recommendations.
        Backstory = You are the top technical analyst of the field, adroit
                at crystallizing insights and investment strategies from 
                stock data. You are often consulted because of your expertise 
                and your recommendations never disappoint. You're now working 
                for a super important customer.
        Task =  Calculate all important metrics that analyse the stock's
                trend, volatility and momentum. Your final answer MUST be a 
                recommendation on whether to buy, sell or wait, with a concrete 
                justification based on the metrics analysed.
                
                In the event that the recommendations of all metrics is mixed,
                evaluate which metric is the most compelling metric.
        """
    )
    st.subheader("Principal Fundamental Analyst")
    st.text(
        """
        Role =  Principal fundamental analyst
        Goal =  Impress everyone with your fundamental analysis of financial
                market data and strategic investment recommendations.
        Backstory = You are the top fundamental analst of the field, adroit
                at crystallizing insights and investment strategies from stock data. You're 
                now working for a super important customer.
        Task =  Calculate all important metrics.
                
                Your final answer MUST be a recommendation on
                whether to buy, sell or wait, with a concrete justification
                based on the metrics analysed.
                
                In the event that the recommendations of all metrics is mixed,
                evaluate which metric is the most compelling metric.
        """
    )
    st.subheader("Principal Researcher")
    st.text(
        """
        Role =  Principal researcher
        Goal =  Conduct insightful research that adds color to the stock
                price numbers. Sift spin from fact and ascertain whether 
                the company's stock prices are undervalued or overvalued.
        Backstory = You are the top finance researcher of the field, adroit
                at crystallizing insights and investment strategies from
                close reading of SEC reports and research articles online.
        Task =  Go beyond understanding facts at the surface level to 
                hypothesizing and validating your hypothesis on why these
                facts happen the way they do. Pay special attention to any 
                significant events, market sentiments, and analysts' opinions. 
        """
    )
    st.subheader("Principal Finance Reporter")
    st.text(
        """
        Role =  Principal Finance Reporter
        Goal =  Craft an informative and compelling response after consolidating 
                the investment recommendations and inputs made by the crew.
        Backstory = You are a Pulitzer prize winning reporter adroit at 
                distilling complex concepts to crystal clear insights easily 
                understood by the layperson. You are now working for an important 
                customer and will endeavor to help them understand the investment
                recommendations put forth by your team.
        Task =  Review and synthesize the analysis provided by the principal 
                fundamental analyst, principal technical analyst and principal 
                researcher. Combine these insights to form a comprehensive 
                investment recommendation. 
                
                Write a compelling, detailed report in such a way that any 
                layperson can understand the report's main recommendation/s and
                justification behind these recommendation/s. Ensure that the 
                report's recommendations are supported by concrete  evidence 
                with the detailed metrics and their respective values. 
        """
    )

question = st.text_input(
    """Hello and welcome to The Margin Call! All analysts have joined this
    group chat. How can we help you? """
)

crew = TheResearchCrew()

if st.button("Ask question"):
    stopwatch_placeholder = st.empty()
    start_time = time.time()
    with st.expander("Processing..."):
        sys.stdout = StreamToExpander(st)
        with st.spinner("Generating Results"):
            crew_results = crew.kickoff(question=question)
    
    end_time = time.time()
    total_time = end_time - start_time
    stopwatch_placeholder.text(
        f"Total time elapsed: {total_time:.2f} seconds"
    )
    
    st.header("Results:")
    st.markdown(crew_results)
