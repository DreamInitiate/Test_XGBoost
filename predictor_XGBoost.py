# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st  

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib  

# 导入 NumPy 库，用于数值计算
import numpy as np  

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd  

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap  

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt  

# 从 LIME 库中导入 LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer  

# 加载训练好的模型（.pkl）
model = joblib.load('xgb_model.pkl')  

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_excel('data.xlsx')  

# 定义特征名称，对应数据集中的列名
feature_names = [  
    "Age",       # 年龄  
    "CE",       # 脑水肿  
    "HT",        # 出血转化  
    "SAP",  # 卒中相关性肺炎  
    "END",      # 早期神经功能恶化
    "Current_Smoking",   # 是否吸烟  
    "NIHSS0",       # 基线NIHSS评分  
    "Operation_time",   # 手术时长      
]  

# Streamlit 用户界面
st.title("AIS patient 90-day prognosis prediction tool:")  # 设置网页标题

# 年龄（Age）：数值输入框
Age = st.number_input("Age:", min_value=0, max_value=120, value=0)  

# 脑水肿（CE）：分类选择框（0：否，1：是）
CE = st.selectbox("Cerebral edema:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  

# 出血转化（HT）：分类选择框（0：否，1：是）
HT = st.selectbox("Hemorrhagic transformation:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  

# 卒中相关性肺炎（SAP）：分类选择框（0：否，1：是）
SAP = st.selectbox("Stroke-associated pneumonia:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  

# 早期神经功能恶化（END）：分类选择框（0：否，1：是）
END = st.selectbox("Early neurological deterioration:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  

# 是否吸烟（Current_Smoking）：分类选择框（0：否，1：是）
Current_Smoking = st.selectbox("Current Smoking:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  

# 基线NIHSS评分（NIHSS0）：数值输入框
NIHSS0 = st.number_input("Baseline NIHSS score:", min_value=0, max_value=42, value=0)  

# 手术时长（Operation_time）：数值输入框
Operation_time = st.number_input("Duration of the surgery:", min_value=0, max_value=1000, value=0) 
 

# 处理输入数据并进行预测
feature_values = [Age, CE, HT, SAP, END, Current_Smoking, NIHSS0, Operation_time]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
     # 预测类别（0：预后良好，1：预后不良）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

     # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of poor prognosis. "
            f"The model predicts that your probability of having a poor outcome within 90 days is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of favorable prognosis. "
            f"The model predicts that your probability of having a favorable outcome within 90 days is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
