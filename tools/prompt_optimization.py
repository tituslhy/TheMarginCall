import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
    FnComponent
)

import dspy
from dspy import Example
from dspy.predict.llamaindex import (
    DSPyComponent,
    LlamaIndexModule
)
from dspy.teleprompt import BootstrapFewShot
from dspy.predict.llamaindex import DSPyPromptTemplate

import sys
__curdir__ = os.getcwd()
if ("tools" in __curdir__) or ("notebooks" in __curdir__):
    sys.path.append(os.path.join(
        __curdir__,
        "../src"
    ))
    storage_dir = "../VectorIndex/"

else:
    sys.path.append("./src")
    storage_dir = "VectorIndex"
    
from utils import CustomWebPageReader
from llamaindex_config import llm, embed_model

### Define variables needed ###
_ = load_dotenv(find_dotenv())

bedrock = dspy.Bedrock(region_name="us-west-2")
lm = dspy.AWSAnthropic(bedrock, 
                       "anthropic.claude-3-haiku-20240307-v1:0")
dspy.settings.configure(lm=lm)
llm = llm
embed_model = embed_model
links = [
    "https://www.investopedia.com/terms/s/stockmarket.asp",
    "https://www.investopedia.com/ask/answers/difference-between-options-and-futures/",
    "https://www.investopedia.com/financial-edge/0411/5-essential-things-you-need-to-know-about-every-stock-you-buy.aspx",
    "https://www.investopedia.com/articles/fundamental/04/063004.asp",
    "https://www.investopedia.com/terms/t/technicalanalysis.asp"  
]
evaluator = SemanticSimilarityEvaluator(similarity_threshold=0.5, embed_model=embed_model)

### Define utility functions ###
class GenerateAnswer(dspy.Signature):
    """Answers questions with short factoid answers."""
    
    context_str = dspy.InputField(desc="contains relevant facts")
    query_str = dspy.InputField()
    answer = dspy.OutputField(desc = "Often between 3 to 5 sentences.")
    
def load_index(persist_dir=storage_dir, 
               links=links,
               embed_model=embed_model):
    """Helper function to create an index from data, persist an index 
    and load an index from storage"""
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(
            persist_dir = storage_dir 
        )
        return load_index_from_storage(storage_context,**{"embed_model":embed_model})
    
    docs = CustomWebPageReader(
        html_to_text=True
    ).load_data(urls=links)
    index = VectorStoreIndex.from_documents(docs,
                                            embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_dir)
    return index

### Prompt Optimization Metric ###
def validate_context_and_answer(example, pred, trace=None):
    """The metric used to validate query engine results. Used as the
    optimization metric for prompt opetimization"""
    result = evaluator.evaluate(
        response = pred.answer,
        reference=example.answer
    )
    return result.passing

### Main function to execute ###
def main():
    """Main function used to train a text_qa_template for a query engine"""
    
    index = load_index()
    retriever = index.as_retriever(similarity_top_k = 2)
    dspy_component = DSPyComponent(
        dspy.ChainOfThought(GenerateAnswer)
    )
    retriever_post = FnComponent(
        lambda contexts: "\n\n".join([n.get_content() for n in contexts])
    )
    p = QP(verbose=True)
    p.add_modules(
        {
            "input": InputComponent(),
            "retriever": retriever,
            "retriever_post": retriever_post,
            "synthesizer": dspy_component,
        }
    )
    p.add_link("input", "retriever")
    p.add_link("retriever", "retriever_post")
    p.add_link("input", "synthesizer", dest_key="query_str")
    p.add_link("retriever_post", "synthesizer", dest_key="context_str")

    dspy_qp = LlamaIndexModule(p)
    
    train_examples = [
        Example(
            query_str = "What is the difference between an option and a future?",
            answer = """An option gives the buyer the right, but not the obligation, to buy (or sell) an asset at a specific price at any time during the life of the contract.
            A futures contract obligates the buyer to purchase a specific asset, and the seller to sell and deliver that asset, at a specific future date
            """
        ),
        Example(
            query_str = "What are the similarities between an option and a future?",
            answer = """Futures and options positions may be traded and closed ahead of expiration, but the parties to the futures contracts for commodities are typically obligated to make and accept deliveries on the settlement date."""
        ),
        Example(
            query_str = "What is an option?",
            answer = "Options are financial derivatives. An options contract gives an investor the right to buy or sell the underlying instrument at a specific price while the contract is in effect. Investors may choose not to exercise their options. Option holders do not own the underlying shares or enjoy shareholder rights unless they exercise an option to buy stock."
        ),
        Example(
            query_str = "What are the different options?",
            answer = "There are only two kinds of options: Call options and put options. A call option confers the right to buy a stock at the strike price before the agreement expires. A put option gives the holder the right to sell a stock at a specific price."
        ),
        Example(
            query_str="What's P/E ratio?",
            answer = """ This ratio is used to measure a company's current share price relative to its per-share earnings. The company can be compared to other, similar corporations so that analysts and investors can determine its relative value. So if a company has a P/E ratio of 20, this means investors are willing to pay $20 for every $1 per earnings. That might seem expensive but not if the company is growing fast. The P/E can be found by comparing the current market price to the cumulative earnings of the last four quarters."""
        ),
        Example(
            query_str="What's a dividend?",
            answer = """Dividends are like interest in a savings account—you get paid regardless of the stock price. Dividends are distributions made by a company to its shareholders as a reward from its profits. The amount of the dividend is decided by its board of directors and are generally issued in cash, though it isn't uncommon for some companies to issue dividends in the form of stock shares."""
        ),
        Example(
            query_str="What's a balance sheet?",
            answer=" A balance sheet is a financial statement that reports a company's assets, liabilities and shareholder equity at a specific point in time"
        ),
        Example(
            query_str="What's a current ratio?",
            answer = "It's the total current assets divided by total current liabilities, commonly used by analysts to assess the ability of a company to meet its short-term obligations"
        ),
        Example(
            query_str = "What are stocks?",
            answer = "When you buy a stock or a share, you're getting a piece of that company. Owning shares gives you the right to part of the company's profits, often paid as dividends, and sometimes the right to vote on company matters"
        ),
        Example(
            query_str = "What are REITs?",
            answer="Real estate investment trusts (REITs) are companies that own, manage, or finance real estate. Investors can buy shares in them, and they legally must provide 90% of their profits as dividends each year."
        ),
        Example(
            query_str = "What are brokers?",
            answer = "Brokers in the stock market play the same role as in insurance and elsewhere, acting as a go-between for investors and the securities markets. They are licensed organizations that buy and sell stocks and other securities for individual and institutional clients."
        ),
        Example(
            query_str = "What is technical analysis?",
            answer = "Technical analysis is used to scrutinize the ways supply and demand for a security affect changes in price, volume, and implied volatility. It assumes that past trading activity and price changes of a security can be valuable indicators of the security's future price movements when paired with appropriate investing or trading rules."
        ),
        Example(
            query_str="What is the difference between fundamental and technical analysis?",
            answer = "Fundamental analysis is a method of evaluating securities by attempting to measure the intrinsic value of a stock. Fundamental analysts study everything from the overall economy and industry conditions to the financial condition and management of companies. Technical analysis differs from fundamental analysis in that the stock's price and volume are the only inputs. The core assumption is that all publicly known fundamentals have factored into price; thus, there is no need to pay close attention to them. Technical analysts do not attempt to measure a security's intrinsic value, but instead, use stock charts to identify patterns and trends that suggest how a stock's price will move in the future."
        )
    ]
    train_examples = [t.with_inputs("query_str") for t in train_examples]
    
    teleprompter = BootstrapFewShot(
        max_labeled_demos=0,
        metric=validate_context_and_answer
    )
    compiled_dspy_qp = teleprompter.compile(dspy_qp, trainset=train_examples)
    qa_prompt_tmpl = DSPyPromptTemplate(
        compiled_dspy_qp.query_pipeline.module_dict["synthesizer"].predict_module
    )
    with open("prompt.txt", "w") as file:
        file.write(qa_prompt_tmpl.get_template())
        file.close()

if __name__ == "__main__":
    main()
    print("Execution complete. Optimized prompt saved to 'prompt.txt'")
