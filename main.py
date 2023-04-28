from Classes import Spell_Checker
from spelling_confusion_matrices import error_tables

path_data = './big.txt'
with open(path_data, 'r') as f:
  text = f.read()
# context = ' '.join(text.split()[5:8]).lower()
# text = text[:4000]
# print(text)
#
#
# # a = Spell
# # _Checker()
# # LM = Language_Model()
# # LM.build_model(text)
# # print('--------------')
# # print(LM.generate())
# # word = 'dino'
#
SP = Spell_Checker()
LM = SP.Language_Model()
LM.build_model(text)
# print('zest' in list(LM.vocabulary))
SP.add_language_model(LM)
SP.add_error_tables(error_tables)
# print(LM.evaluate_text('I had seen little of Holmes lately.'))
# print(LM.evaluate_text('I had seen little dino of Holmes .'))
print(SP.spell_check('I had seen litle of Holmes lately. My mgrriaae had drifted us away from each other.',
                     alpha=0.95))

