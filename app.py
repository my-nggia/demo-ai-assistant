import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from processing_data import ProcessingModel
from questions_classification import QuestionClassifierModel

st.title("Chat")
st.markdown("Author: my.ngngia@gmail.com")

conn = st.connection("gsheets", type=GSheetsConnection)

def fetch_data(): 
    return conn.read(worksheet="TransactionData", usecols=list(range(11)), ttl=5)

df = fetch_data()
data_csv = pd.DataFrame(df)
old_data = pd.read_csv("cleaned_online_retail.csv")

data = pd.concat([old_data, data_csv])

processing_model = ProcessingModel(data=data)


def action(prompt):
    categories = ["Product Analysis", "Sales Analysis", "Customer Analysis", "RFM Clustering"]
    if predicted_category == categories[0]:
        if "best" in prompt.lower(): 
            print("-- Best Product --")
            return st.table(processing_model.get_top_products(False))
        elif "worst" in prompt.lower(): return st.table(processing_model.get_top_products(True))
        else: return st.table(processing_model.get_top_products(False))
    
    elif predicted_category == categories[1]:
        return "Not supported yet."
    
    elif predicted_category == categories[2]:
        if "best" in prompt.lower(): 
            return st.table(processing_model.get_top_customers(False))
        elif "worst" in prompt.lower(): return st.table(processing_model.get_top_customers(True))
        else: return st.table(processing_model.get_top_customers(False))
        
    elif predicted_category == categories[3]:
        return "RFM" 

def promt_classification(predicted_category, promt):
    categories = ["Product Analysis", "Sales Analysis", "Customer Analysis", "RFM Clustering"]
    if predicted_category == categories[0]:
        return 0
    
    elif predicted_category == categories[1]:
        return 1
    
    elif predicted_category == categories[2]:
        return 2
    
    elif predicted_category == categories[3]:
        return 3
    
    else: "I'm sorry, I didn't understand your questions. Can you rephrase it?"
    

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    classifier = QuestionClassifierModel.load_model("question_classifier.pkl")
    predicted_category = classifier.predict(prompt)
    
    print(prompt, " | ", predicted_category ,"\n")
    
    
    response = promt_classification(predicted_category, prompt)
    # action = action(prompt)
    
    with st.chat_message("assistant"):
        if response == 0:
            if "best" in prompt.lower(): 
                st.table(processing_model.get_top_products(False))
            elif "worst" in prompt.lower(): st.table(processing_model.get_top_products(True))
            else: st.table(processing_model.get_top_products(False))
            
        elif response == 1:
            st.markdown("This function is not supported yet.")
            
        elif response == 2: 
            if "best" in prompt.lower(): 
                st.table(processing_model.get_top_customers(False))
            elif "worst" in prompt.lower(): st.table(processing_model.get_top_customers(True))
            else: st.table(processing_model.get_top_customers(False))    
        elif response == 3: 
            elbow_data = pd.DataFrame(
                {
                    'x': range(1, 11),
                    'y': processing_model.rfm()
                }
            )
            st.line_chart(elbow_data.set_index('x'))
               
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": action})
        

