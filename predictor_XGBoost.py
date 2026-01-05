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
model = joblib.load('random_forest_model.pkl')  

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
st.title("90天预后预测工具")  # 设置网页标题

# 年龄（Age）：数值输入框
Age = st.number_input("年龄:", min_value=0, max_value=120, value=0)  

# 脑水肿（CE）：分类选择框（0：否，1：是）
CE = st.selectbox("脑水肿:", options=[0, 1], format_func=lambda x: "否" if x == 1 else "是")  

# 出血转化（HT）：分类选择框（0：否，1：是）
HT = st.selectbox("出血转化:", options=[0, 1], format_func=lambda x: "否" if x == 1 else "是")  

# 卒中相关性肺炎（SAP）：分类选择框（0：否，1：是）
SAP = st.selectbox("卒中相关性肺炎:", options=[0, 1], format_func=lambda x: "否" if x == 1 else "是")  

# 早期神经功能恶化（END）：分类选择框（0：否，1：是）
END = st.selectbox("早期神经功能恶化:", options=[0, 1], format_func=lambda x: "否" if x == 1 else "是")  

# 是否吸烟（Current_Smoking）：分类选择框（0：否，1：是）
Current_Smoking = st.selectbox("是否吸烟:", options=[0, 1], format_func=lambda x: "否" if x == 1 else "是")  

# 基线NIHSS评分（NIHSS0）：数值输入框
NIHSS0 = st.number_input("基线NIHSS评分:", min_value=0, max_value=42, value=0)  

# 手术时长（Operation_time）：数值输入框
Operation_time = st.number_input("手术时长（分钟数）:", min_value=0, max_value=1000, value=0) 
 

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

    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(X_test)
    
    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not poor outcome', 'poor outcome'],  # Adjust class names to match your classification task
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)