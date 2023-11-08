import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemagent.tools.databases import *
from chemagent.tools.search import *
from chemagent.tools.rdkit import *
from chemagent.tools.models import *
from chemagent.tools.interaction import *

def make_tools(llm: BaseLanguageModel, verbose=False):
    # serp_key = os.getenv("SERP_API_KEY")

    # all_tools = agents.load_tools(["python_repl", "human"])
    # all_tools = agents.load_tools()
    # all_tools = agents.load_tools(["interaction"])

    all_tools = [
        # Query2SMILES(),
        # Query2CAS(),
        # QueryUSPTOWithType(), # TODO 导入数据
        # ReactionT5(),
        NERF_non_reactant_mask(),
        NERF_know_reagents(),
        # PatentCheck(),
        # MolSimilarity(),
        # SMILES2Weight(),
        FuncGroups(),
        # LitSearch(llm=llm, verbose=verbose), # TODO debug
        # HumanInputRun()
    ]
    # if serp_key:
    #     all_tools.append(WebSearch())
    return all_tools