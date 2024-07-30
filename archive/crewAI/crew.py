from crewai import Crew, Process
from crew_agents import (
    data_analyst,
    technical_analyst,
    fundamental_analyst,
    professor,
    reporter,
    manager
)
from crew_tasks import StockAnalysisTasks

class StockCrew:
    # data_analyst = data_analyst
    technical_analyst = technical_analyst
    fundamental_analyst = fundamental_analyst
    professor = professor
    reporter = reporter
    manager = manager
    
    def __init__(self, 
                 company: str):
        self.company = company
        
        ## Instantiate tasks
        tasks = StockAnalysisTasks()
        fa_task = tasks.fundamental_analysis(
            agent = self.fundamental_analyst, 
            company = self.company)
        ta_task = tasks.technical_analysis(
            agent = self.technical_analyst, 
            company = self.company)
        # da_task = tasks.data_analysis(
        #     agent = self.data_analyst, 
        #     company =self.company)
        review_task = tasks.review(agent = self.professor)
        report_task = tasks.report(agent = self.reporter)

        ## Instantiate crew
        self.crew = Crew(
            agents = [
                # self.data_analyst,
                self.technical_analyst,
                self.fundamental_analyst,
                self.professor,
                self.reporter
            ],
            tasks = [
                # da_task,
                fa_task,
                ta_task,
                review_task,
                report_task
            ],
            manager_agent = self.manager,
            process = Process.hierarchical
        )
        
    def run(self):
        return self.crew.kickoff()