import sys
sys.path.append('..')
from chemagent import *
from chemagent.agents.chem_agent import ChemAgent
import sys
sys.path.append('../chemagent/tools/NERF')
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# os.environ["SERPER_API_KEY"] = "your-serper-key"

import pandas as pd

if __name__ == '__main__':
    model = ChemAgent(model='gpt-4-0613', analysis_model='gpt-4-0613', max_iterations=10)

    sample_data = pd.read_csv('fg_sample_data_0911_200.csv').iloc[0:5]
    # sep_function_data = pd.read_csv('../data/uspto_sep.csv') # HERE ===

    result_list = []
    # for index, row in sample_data1.iterrows():
    #     reactants = row['reactants']
    #     reagents = row['reagents']
    #     result = model.run(f'Now You are given the reactants: {reactants}, and reagents: {reagents}, \
    #            predict the possible product SMILES. Meanwhile, explain the reaction mechanism.')
    #     result_list.append({'reactants': reactants, 'reagents': reagents, 'result': result[0]})

    for index, row in sample_data.iterrows():
        reactants = row['reactants']
        reagents = row['reagents']
        try:
            # result = model.run(f'Now You are given the reactants: {reactants}, and reagents: {reagents}. \
            #        There are 4 tasks: \
            #        Question1. What are the reactants after atom mapping?\
            #        Question2. predict the possible product SMILES (both in atom-mapping style and non-atom-mapping style). \
            #        Question3. analyze the changes of functional groups during the reaction, \
            #        Question4. Which atoms in reactants are the reactive center? \
            #        (Format of Question4: reactants after atom mapping are: ...  The reactive centers are atom 11 (C element), atom 12 (O element), and atom 15 (C element).)')

            result = model.run(f'Now You are given the reactants: {reactants}, and reagents: {reagents}. \
                   There are 2 tasks: \
                   Question1. predict the possible product SMILES. \
                   Question2. analyze the changes of functional groups during the reaction.')
            result_list.append({'reactants': reactants, 'reagents': reagents, 'result': result[0]})
        except Exception as e:
            print(f"An error occurred for index {index}: {e}")
            continue

    result = pd.DataFrame(result_list) # HERE ===
    result.to_csv('f_sample_agent_result_0911_5.csv', index=False) # HERE ===
