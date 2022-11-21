

#%%
from deep_translator import GoogleTranslator
keywords_df = pd.read_csv('.\\INPUT_DATA\\KWR_Thomas.csv', sep=',')

df= data_preprocessing(keywords_df)