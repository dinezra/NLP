from Classes import Spell_Checker
from Data.spelling_confusion_matrices import error_tables

path_data = 'Data/big.txt'
with open(path_data, 'r') as f:
  text = f.read()

SP = Spell_Checker()
LM = SP.Language_Model()
LM.build_model(text)

SP.add_language_model(LM)
SP.add_error_tables(error_tables)
print(SP.spell_check('I had seen litle of Holmes lately.',
                     alpha=0.95))

