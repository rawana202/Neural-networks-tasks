import streamlit as st
from Data.DataPreprocessing  import DataPreprocessing
from SLP.testing import SLPTesting
from SLP.training import SLPTraining

st.set_page_config(
    page_title="Neural Network Tasks",
    layout="wide"
)

st.set_page_config(
    page_title="Neural Network Tasks",
    layout="wide"
)

st.title("Neural Network Tasks")
st.divider()

tab1, tab2 = st.tabs(
    ["TaskOne", "TaskTwo"]
)

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.subheader("Choose the target classes")
            classes = ["Adelie & Chinstrap", "Chinstrap & Gentoo", "Adelie & Gentoo"]
            target_classes = st.selectbox(
            "Select two classes:",
            options=classes
           )
    
    with col2:
        with st.container(border=True):
            st.subheader("Choose the input features")
            features = ["CulmenLength & CulmenDepth", "CulmenLength & FlipperLength", "CulmenLength & OriginLocation", "CulmenLength & BodyMass",
                        "CulmenDepth & FlipperLength", "CulmenDepth & OriginLocation", "CulmenDepth & BodyMass",
                        "FlipperLength & OriginLocation", "FlipperLength & BodyMass", "OriginLocation & BodyMass"]
                       
            input_features = st.selectbox(
            "Select two features:",
            options=features
            )

            features = input_features.split(" & ")
        
    with col3:
        with st.container(border=True):  
            st.subheader("Add Num Of Learning Rate (eta)") 
            lr = st.number_input(
                "Enter a number",
                min_value=0.0,
                step=0.1,
                value=0.1
                )

    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):  
            st.subheader("Select the model type") 
            models = ["SLP", "Adaline"]
            model = st.radio(
                "model type",
                options=models,
                horizontal=True,
                index=None
                )
            
            st.subheader("Add Bias(b)")
            choices = ["Yes", "No"]
            bias = st.radio(
                "bias",
                options=choices,
                horizontal=True,
                index=None
                )
            
    with col5:
        
        x_train, y_train, x_test, y_test = DataPreprocessing(features, target_classes)

        if model == "SLP":
            with st.container(border=True):  
                st.subheader("Add Num Of Epochs") 
                epochs = st.number_input(
                    "Enter a epoc number",
                    min_value=1,
                    step=1,
                    value=10
                    )

            w0, w1, w2 = SLPTraining(x_train, y_train, lr, epochs, bias)  
            accuracy = SLPTesting(x_test, y_test, w0, w1, w2)      
            
        else:
            with st.container(border=True):  
                st.subheader("Add MSE Threshold ") 
                mse_threshold = st.number_input(
                    "Enter mse_threshold",
                    min_value=0.0,
                    step=0.1,
                    value=0.1
                    )        


                

    with col6:
        if model == "SLP":
            st.subheader("SLP Model Results")
            st.write(f"Final Weights: w0={w0}, w1={w1}, w2={w2}")
            st.write(f"Accuracy: {accuracy:.2f}%")
        else:
            st.subheader("Adaline Model Results")
            st.write(f"MSE Threshold: {mse_threshold}")

                
