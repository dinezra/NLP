from Classes import Spell_Checker
from Data.spelling_confusion_matrices import error_tables

path_data = 'Spell Checker/Data/big.txt'
with open(path_data, 'r') as f:
  text = f.read()

SP = Spell_Checker()
LM = SP.Language_Model()
LM.build_model(text)

SP.add_language_model(LM)
SP.add_error_tables(error_tables)
print(SP.spell_check('din',
                     alpha=0.95))

