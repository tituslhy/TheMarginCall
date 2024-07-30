from crewai import Task
from textwrap import dedent
from typing import Union, List

class StockAnalysisTasks:
    """Collection of tasks for the stock analysis crew"""
    
    def fundamental_analysis(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Undertake fundamental analysis by analysing the stock price data
                to answer the question: {question}
                
                Calculate all important metrics.
                
                Your final answer MUST be a recommendation on
                whether to buy, sell or wait, with a concrete justification
                based on the metrics analysed.
                
                In the event that the recommendations of all metrics is mixed,
                evaluate which metric is the most compelling metric.
                
                {self.__tip_section()}
                """),
            agent = agent,
            expected_output = """A comprehensive, compelling analysis report including an investment
            recommendation and concrete justification with details of the metrics analysed
            and their respective values.
            """
        )
    
    def technical_analysis(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Undertake technical analysis by analysing the stock price data
                to answer the question: {question}.
                
                Calculate all important metrics that analyse the stock's
                trend, volatility and momentum. Your final answer MUST be a recommendation on
                whether to buy, sell or wait, with a concrete justification
                based on the metrics analysed.
                
                In the event that the recommendations of all metrics is mixed,
                evaluate which metric is the most compelling metric.
                
                {self.__tip_section()}
                """
            ),
            agent = agent,
            expected_output = """A comprehensive, compelling analysis report including an investment
            recommendation and concrete justification with details of the metrics analysed
            and their respective values.
            """
        )
    
    def data_analysis(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Undertake data analysis by analysing the stock price data  to answer the question: 
                {question}
                
                Identify important trends that are not apparent within the data such as rolling 
                averages, computing compund annual growth rate (cagr), getting descriptive
                statistics, etc. Your final answer MUST be a report summarizing your analysis
                of the data as it will be used for decision-making.
                
                {self.__tip_section()}
                """
            ),
            agent = agent,
            expected_output = """A comprehensive, compelling analysis report with details of the
            your analysis.
            """
        )
    
    def explain_concepts(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Answer technical definitions asked by the user and help to clarify any conceptual
                misunderstandings. 
                
                This is the user's question: {question}
                
                {self.__tip_section()}
                """
            ),
            agent = agent,
            expected_output = """A crisp answer that helps the layman understand the mysterious
            underpinnings of the stock market."""
        )
    
    def research(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Undertake close reading of the relevant SEC reports and
                online research to answer the question: {question}
                
                Go beyond understanding facts at the surface level to hypothesizing and
                validating your hypothesis on why these facts happen the way they do.
                Pay special attention to any significant events, market sentiments, and
                analysts' opinions. 
                
                {self.__tip_section()}
                """
            ),
            agent = agent,
            expected_output ="""Your final answer MUST be a report that incldues a
            comprehensive summary of the latest news, insights from the SEC reports,
            and potential impacts on the stock."""
        )
    
    def report(self, agent, question: str):
        return Task(
            description = dedent(
                f"""
                Review and synthesize the analysis provided by the principal fundamental 
                analyst, principal technical analyst and principal researcher. Combine these
                insights to form a comprehensive investment recommendation. 
                
                Write a compelling, detailed report in such a way that any 
                layperson can understand the report's main recommendation/s and justification behind
                these recommendation/s.
                
                The report must answer the question: {question}
                
                Ensure that the report's recommendations are supported by concrete 
                evidence with the detailed metrics and their respective values. 
                
                {self.__tip_section()}
                """
            ),
            agent = agent,
            expected_output = """A compelling report worthy of the Pulitzer prize."""
        )
    
    def __tip_section(self):
        """Just a little extra motivation for the agents"""
        return "If you do your BEST WORK, I'll give you a $10,000 commission!" 