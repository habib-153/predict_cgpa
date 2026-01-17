import gradio as gr
import numpy as np
import pandas as pd
import pickle

#load model
with open('student_performance_model.pkl', 'rb') as f:
    model = pickle.load(f)

#main logic


def predict_gpa(gender, age, address, famsize, Pstatus, M_Edu, F_Edu, M_Job, F_Job, relationship, smoker, tuition_fee, time_friends, ssc_result):
    input_df = pd.DataFrame([
        [
            gender,age,address,famsize,Pstatus,M_Edu,F_Edu,M_Job,F_Job,relationship,smoker,tuition_fee,time_friends,ssc_result
        ]
    ],
    columns =[
        'gender','age','address','famsize','Pstatus','M_Edu','F_Edu','M_Job','F_Job','relationship','smoker','tuition_fee','time_friends','ssc_result'
    ]
    )

    prediction = model.predict(input_df)
    return f"Predicted HSC Result (GPA): {np.clip(prediction[0], 0, 5):.2f}"

inputs = [
    gr.Dropdown(choices=['M', 'F'], label='Gender'),
    gr.Number(label='Age'),
    gr.Dropdown(choices=['Urban', 'Rural'], label='Address'),
    gr.Dropdown(choices=['LE3', 'GT3'], label='Family Size'),
    gr.Dropdown(choices=['Together', 'Apart'], label='Parental Status'),
    gr.Slider(0, 4, step=1, label='Mother Education Level'),
    gr.Slider(0, 4, step=1, label='Father Education Level'),
    gr.Dropdown(choices=['Teacher', 'Health', 'Services',
                'At_home', 'Other'], label='Mother Job'),
    gr.Dropdown(choices=['Teacher', 'Health', 'Services',
                'Farmer', 'Business'], label='Father Job'),
    gr.Dropdown(choices=['mother', 'father', 'other'], label='Guardian'),
]
#interface
app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs=gr.Textbox(label="Greeting"),
    title="Bangladesh Student Performance Predictor",
    description="Predict the HSC result (GPA) of Bangladeshi students based on various demographic and academic features."
)

#launch
app.launch()