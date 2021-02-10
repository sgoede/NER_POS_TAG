import streamlit as st, base64, pandas as pd, time
from flair.data import Sentence
from flair.models import MultiTagger
from io import BytesIO
from streamlit import caching

# load tagger for POS and NER
tagger = MultiTagger.load(['ner-fast','pos-fast'])

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer,sheet_name='NER&POS', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_tabe_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    link = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download = "output.xlsx"><input type = "button" value = "Click here to download the tagged dataset"></a>'
    return link

def labelmaker(list):
    sentences = []
    tags = []
    for row in range(len(raw)):
        sentence = Sentence(raw[row])
        sentences.append(sentence)
        tagger.predict(sentences[row])
        tags.append(sentence.to_tagged_string())
    return tags

@st.cache(allow_output_mutation=True)
def load_data(file):
    dataframe_raw = pd.read_excel(file, keep_default_na=False).fillna('')
    return dataframe_raw

st.title("English Named Entity Recognition (NER) and Part-of-Speech (POS) tagger")
st.subheader("Created by: Stephan de Goede")

use_type = st.sidebar.radio("Please select the type of input you want to use",
                             ("Manually type a sentence","Automatically tag an Excel file"))

if use_type == "Manually type a sentence":
    st.title("Please type something in the box below")
    manual_user_input = st.text_area("")
    if len(manual_user_input)>0:
        manual_sentence = Sentence(manual_user_input)
        tagger.predict(manual_sentence)
        tagged_string = manual_sentence.to_tagged_string()
        st.success("Below is your tagged string.")
        st.write(tagged_string)

elif use_type == "Automatically tag an Excel file":
    st.title("Please select your Excel to have it tagged")
    user_input = st.file_uploader("None of your data will be copied. Please be responsible and do not upload sensitive data.",type=['xlsx'],key='file_uploader')
    if user_input is not None:
        data_load_state = st.text("Loading data...")
        dataframe = load_data(user_input)
        data_load_state.text("Loading data...done!")
        st.write(dataframe)
        target = st.selectbox('Please select the column that you want to TAG', options=dataframe.columns)
        if st.button(f"You have selected: {target}. Please click here to confirm"):
            raw = dataframe[target].astype(str).to_list()
            output = labelmaker(raw)
            dataframe["tagging_result"] = output
            button = st.empty()
            button.markdown(get_tabe_download_link(dataframe),unsafe_allow_html= True)
            with st.spinner(" Please download your file in the next 3 minutes, after this period it is deleted"):
                st.info("Trouble downloading? Please use Chrome or Firefox right click on the button and select: save as")
                time.sleep(180)
            caching.clear_cache()
            button.empty()
            st.success("The file is deleted")