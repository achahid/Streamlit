

#streamlit run main.py

import streamlit as st
import nltk
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import sys
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util





def data_preprocessing(df):
    # check if KEYWORD column does exists:
    if 'Keyword' not in df.columns:
        sys.exit("Keyword column does not exist in the data or is misspelled."
                 "consider fixing this error and try it again ")
    # data translate:
    df.dropna(inplace=True)
    print("transliting")
    df["Keyword_ENG"] = df["Keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
    # remove question related words, otherwise K-MNEANS will cluster them as a cluster:
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'what is', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'why is', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'what', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'why', '')
    df['id'] = range(len(df))
    df = df[['id','Keyword','Keyword_ENG']]
    # col = df.pop("id")
    # df.insert(0, col.name, col)

    return (df)


def stemmList(list):
    # the stemmer requires a language parameter
    snow_stemmer = SnowballStemmer(language='english')
    # porter_stemmer = PorterStemmer()

    nltk.download('punkt')
    stemmed_list = []
    for l in list:
        words = l.split(" ")
        stem_words = []
        # print(l)
        for word in words:
            x = snow_stemmer.stem(word)
            # x = porter_stemmer.stem(word)
            stem_words.append(x)
        key = " ".join(stem_words)
        # print(key)
        stemmed_list.append(key)
    return stemmed_list


def cluster_label(df, cluster_num):
    'Function to lable each cluster correctly based on word frequencies'
    x = df[df["cluster"] == cluster_num]
    a = x['Keyword'].str.split(expand=True).stack().value_counts()
    A = pd.DataFrame(a)
    A['lables'] = A.index
    A.rename(columns={0: 'Values'}, inplace=True)
    # indSort = np.argsort(A.Values)[::-1]
    A.reset_index(drop=True, inplace=True)
    return A


def clustering(data, Keyword_ENG, start_cluster, end_cluster, steps, cutoff):
    textlist = Keyword_ENG.to_list()
    textlist = stemmList(textlist)
    text_data = pd.DataFrame(textlist)
    # Bag of words
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(textlist)

    dic = {}

    for cl_num in range(start_cluster, end_cluster, steps):

        try:
            kmeans = KMeans(n_clusters=cl_num, random_state=10)
            kmeans.fit(X_cv)
            result = pd.concat([text_data, pd.DataFrame(X_cv.toarray(), columns=vectorizer_cv.get_feature_names_out())],
                               axis=1)
            result['cluster'] = kmeans.predict(X_cv)
            result.rename(columns={0: 'Keyword'}, inplace=True)
            df_results = result[['Keyword', 'cluster']].copy()
            df_results['id'] = data.id.values

            df_matched = pd.merge(data, df_results, on="id", how="left")

            df_matched.rename(columns={'Keyword_y': 'Keyword'}, inplace=True)

            # This step is needed to create labels on  automatic way
            # Adding labels to cluster : we will use only one word for topic as label.
            labeled_df = df_matched.copy()
            for num_cl in range(cl_num + 1):
                cl_df = cluster_label(df_matched, num_cl)
                cl_lables = ' '.join(cl_df.lables[:2])  # choose the most frequent
                labeled_df.loc[labeled_df.cluster == num_cl, 'Lables'] = cl_lables

            # STEP: we have to find  meaningful label to the cluster. In the previouse step the lable was based on 'Keyword'
            # which was a stemming variant of 'Keyword_ENG'...

            sentences1 = labeled_df.Lables
            sentences2 = labeled_df.Keyword_ENG
            # sentences2 = labeled_df.Keyword

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            x = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                x.append(cosine_scores[i][i].item())

            labeled_df['Semantic_score'] = x

            dt = labeled_df[ ['id', 'Keyword_x', 'Keyword_ENG', 'cluster', 'Lables', 'Semantic_score']].copy()
            df_0 = dt.groupby('cluster')['Semantic_score'].max().reset_index()
            df_1 = pd.merge(dt, df_0, on="cluster", how="left")
            df_1.rename(columns={'Semantic_score_y': 'high_score', 'Semantic_score_x': 'Semantic_score',
                                 'Keyword_x': 'Keyword'}, inplace=True)
            df_2 = df_1[df_1.Semantic_score == df_1.high_score]
            df_2.drop_duplicates(subset=['Lables', 'Semantic_score', 'high_score'], keep='first', inplace=True)
            df_2.rename(columns={'Keyword_ENG': 'LABELS'}, inplace=True)
            df_2.drop(['Lables', 'id', 'Semantic_score', 'Keyword'], axis=1, inplace=True)

            # df_3 = pd.merge(dt[['id','cluster','Semantic_score']],df_2, on="cluster", how="left")
            df_3 = pd.merge(dt[['id', 'Keyword_x', 'cluster', 'Semantic_score', 'Keyword_ENG']], df_2, on="cluster",
                            how="left")
            df_3.rename(columns={'Keyword_x': 'Keyword'}, inplace=True)
            df_3.drop(['high_score', 'Semantic_score'], axis=1, inplace=True)
            df_3 = df_3[['id', 'Keyword', 'Keyword_ENG', 'cluster', 'LABELS']]

            # Similarity score calculation between LABELS AND KEYWORD ENGLISH.
            # to have an idea how far a keyword from specific cluster.
            sentences1 = df_3.LABELS
            sentences2 = df_3.Keyword_ENG

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            y = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                y.append(cosine_scores[i][i].item())

            df_3['Semantic_score'] = y

            z = df_3.groupby(['cluster'])['Semantic_score'].mean()
            A = z[z < cutoff]
            print("DATA WITH {} CLUSTERS WAS GENERATED. HOWEVER CLUSTERS {} WERE NOISY".format(num_cl, A.index.values))
            # logger.info(  "DATA WITH {} CLUSTERS WAS GENERATED. HOWEVER CLUSTERS {} WERE NOISY".format(num_cl, A.index.values))

            dic[cl_num] = df_3

        except Exception as e:
            print(e)
            continue
    return (dic)


def clustering_sub_topic(data, Keyword_ENG, start_cluster, end_cluster, steps, cutoff):
    textlist = Keyword_ENG.to_list()
    textlist = stemmList(textlist)
    text_data = pd.DataFrame(textlist)
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(textlist)

    dic = {}

    for cl_num in range(start_cluster, end_cluster, steps):

        try:
            # cl_num = 2
            # data = df_2.copy()
            kmeans = KMeans(n_clusters=cl_num, random_state=10)
            kmeans.fit(X_cv)
            result = pd.concat([text_data, pd.DataFrame(X_cv.toarray(), columns=vectorizer_cv.get_feature_names_out())],
                               axis=1)
            result['cluster'] = kmeans.predict(X_cv)
            result.rename(columns={0: 'Keyword'}, inplace=True)
            df_results = result[['Keyword', 'cluster']].copy()
            df_results['id'] = data.id.values

            df_matched = pd.merge(data, df_results, on="id", how="left")
            df_matched.rename(columns={'Keyword_y': 'Keyword'}, inplace=True)

            labeled_df = df_matched.copy()
            for num_cl in range(cl_num + 1):
                cl_df = cluster_label(df_matched, num_cl)
                cl_lables = ' '.join(cl_df.lables[:2])
                # print('\n\n','LABLE NAME ::  ',cl_lables,'\n\n',cl_df.head())

                labeled_df.loc[labeled_df.cluster == num_cl, 'Lables'] = cl_lables

            '''This step is needed to create labels on  automatic way:'''
            sentences1 = labeled_df.Lables
            # sentences2 = labeled_df.Keyword_ENG
            sentences2 = labeled_df.Keyword

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            x = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                x.append(cosine_scores[i][i].item())

            labeled_df['Semantic_score'] = x

            dt = labeled_df[['id',   'Keyword_x', 'Keyword_ENG',
                             'Semantic_score', 'cluster_0', 'LABELS_0', 'Semantic_score_0', 'Lables', 'Keyword'
                , 'cluster']].copy()
            df_0 = dt.groupby('cluster')['Semantic_score'].max().reset_index()
            df_1 = pd.merge(dt, df_0, on="cluster", how="left")
            df_1.rename(columns={'Semantic_score_y': 'high_score', 'Semantic_score_x': 'Semantic_score',
                                 'Keyword_x': 'Keyword'}, inplace=True)
            df_2 = df_1[df_1.Semantic_score == df_1.high_score]
            df_2.drop_duplicates(subset=['Lables', 'Semantic_score', 'high_score'], keep='first', inplace=True)
            df_2.rename(columns={'Keyword_ENG': 'LABELS'}, inplace=True)
            df_2.drop(['Lables', 'id', 'Semantic_score', 'Keyword'], axis=1, inplace=True)

            # df_3 = pd.merge(dt[['id','cluster','Semantic_score']],df_2, on="cluster", how="left")
            df_3 = pd.merge(dt[['id', 'Keyword_x', 'cluster', 'Semantic_score', 'Keyword_ENG']], df_2, on="cluster",
                            how="left")
            df_3.rename(columns={'Keyword_x': 'Keyword'}, inplace=True)
            df_3.drop(['high_score', 'Semantic_score'], axis=1, inplace=True)
            df_3 = df_3[['id',   'Keyword', 'Keyword_ENG', 'cluster_0', 'LABELS_0',
                         'Semantic_score_0', 'cluster', 'LABELS']]

            # new subTOPIC labeling:
            df_3.rename(columns={'LABELS': 'SUB_TOPIC'}, inplace=True)

            # similarity score calculation:
            # sentences1 = df_3.TOPIC_new
            sentences1 = df_3.SUB_TOPIC
            sentences2 = df_3.Keyword_ENG

            # Compute embedding for both lists
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)

            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            y = []
            # Output the pairs with their score
            for i in range(len(sentences1)):
                # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
                y.append(cosine_scores[i][i].item())

            df_3['Semantic_score'] = y
            df_3.rename(columns={'cluster': 'sub_cluster', 'Semantic_score': 'sub_Semantic_score'}, inplace=True)

            z = df_3.groupby(['sub_cluster'])['sub_Semantic_score'].mean()
            A = z[z < cutoff]
            X = len(A.index.values)
            print("NOISY CLUSTERS FOR CLUSTER CUTOFF {} ARE : {}".format(num_cl, A.index.values))
            # logger.info( " CLUSTERS :{} *** ARE NOT COMPACT WHEN USING {} AS CUTOFF LEVEL".format(A.index.values, num_cl))

            dic[cl_num] = df_3
            if X == 0:
                break

        except Exception as e:
            print(e)
            continue
    return (dic)


def re_scoring(df):
    sentences1 = df.LABELS
    sentences2 = df.Keyword_ENG

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    scores = []
    # Output the pairs with their score
    for i in range(len(sentences1)):
        scores.append(cosine_scores[i][i].item())

    df['Semantic_score'] = scores
    z = df.groupby(['cluster'])['Semantic_score'].mean()
    A = z[z < 0.5]
    print("THIS CLUSTERS {} WERE NOISY".format(A.index.values))
    #logger.info("THIS CLUSTERS {} WERE NOISY".format(A.index.values))

    return (df)


def info(df, cutoff):
    z = df.groupby(['cluster'])['Semantic_score'].mean()
    A = z[z < cutoff]
    # x = ("FOR THIS DATA, THE NEXT CLUSTERS {} WERE NOISY".format(A.index.values))
    x = A.index.values
    return (x)





st.set_page_config(page_title='KEYWORDS CLUSTERING')
st.header('KEYWORDS CLUSTERING')

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
 
</style>""", unsafe_allow_html=True)



# Preprocessing the data:
data_preprocessed  = st.checkbox('Select keyword data to pre-process')

if data_preprocessed:
    uploaded_file_pr = st.file_uploader("Upload data for cleaning", type=['csv'])


    if uploaded_file_pr is not None:
        df = pd.read_csv(uploaded_file_pr)
        st.info('Please wait while Data is undergoing a preprocessing phase: Checks, Translation, Cleaning ....')
        df = data_preprocessing(df)
        st.dataframe(df)
        df.to_csv('.\\CLUSTERING_RESULTS\\KEYWORDS_DATA.csv',
                      index=False,
                      sep=',',  # file tipe delimiter  ";" or "," or ....
                      header=True,  # if you want to show the column names.
                      decimal=".",  # replacing the"." as comma in digits.
                      encoding="utf-8",
                      line_terminator='\n',
                      float_format='%.3f'
                      )




pick_data_cl = st.checkbox('Select keyword data to cluster')
# pick_data_cl = st.button('Select keyword data to cluster')


if pick_data_cl:
    uploaded_file_cl = st.file_uploader("Upload data", type=['csv'])



    if uploaded_file_cl is not None:
        df = pd.read_csv(uploaded_file_cl)
        st.dataframe(df)


load = st.button('GENERATE CLUSTERS')
# load = st.checkbox('GENERATE CLUSTERS')




if load:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    Keyword_ENG = df['Keyword_ENG']
    dic = clustering(df, Keyword_ENG, start_cluster=25, end_cluster=125, steps=25, cutoff=0.5)
    for i in dic:

        dic[i].to_csv('.\\CLUSTERING_RESULTS\\CLUSTER_id_' + str(i) + '.csv',
                      index=False,
                      sep=',',  # file tipe delimiter  ";" or "," or ....
                      header=True,  # if you want to show the column names.
                      decimal=".",  # replacing the"." as comma in digits.
                      encoding="utf-8",
                      line_terminator='\n',
                      float_format='%.3f'
                      )
    st.balloons()
    st.success('CLUSTERS WERE SUCCESSFULLY GENERATED')

pick_data = st.checkbox('Select clustered data to validate')

if pick_data:
    uploaded_file = st.file_uploader("Upload a cluster data", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        txt = info(df, cutoff=0.5)
        st.write("Please double-check the following clusters", txt)
