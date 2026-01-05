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
try:
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    
    # 准备输入数据
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(input_data)
    
    # 调试信息（可选）
    with st.expander("SHAP调试信息"):
        st.write(f"Expected value: {explainer_shap.expected_value}")
        st.write(f"Expected value类型: {type(explainer_shap.expected_value)}")
        st.write(f"SHAP values类型: {type(shap_values)}")
        if isinstance(shap_values, list):
            st.write(f"SHAP值列表长度: {len(shap_values)}")
            for i, val in enumerate(shap_values):
                if hasattr(val, 'shape'):
                    st.write(f"SHAP[{i}] 形状: {val.shape}")
                else:
                    st.write(f"SHAP[{i}]: {type(val)}")
        elif hasattr(shap_values, 'shape'):
            st.write(f"SHAP值形状: {shap_values.shape}")
    
    # XGBoost二分类模型的SHAP值处理
    # 对于二分类XGBoost，shap_values通常是一个列表 [shape(1, n_features), shape(1, n_features)]
    # expected_value通常是一个标量或长度为2的数组
    
    # 方法1：简化版本，不区分类别显示
    st.subheader("SHAP Force Plot")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # 处理expected_value
    if isinstance(explainer_shap.expected_value, np.ndarray):
        base_value = explainer_shap.expected_value[1]  # 正类的期望值
    else:
        base_value = explainer_shap.expected_value  # 标量期望值
    
    # 处理shap_values
    if isinstance(shap_values, list):
        # 对于二分类，通常有两个数组，取第二个（正类）
        shap_val = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_val = shap_values
    
    # 确保是二维数组
    if len(shap_val.shape) == 1:
        shap_val = shap_val.reshape(1, -1)
    
    # 创建force plot
    shap.force_plot(
        base_value,
        shap_val[0],  # 取第一个样本
        input_data.iloc[0],  # 特征数据
        matplotlib=True,
        show=False,
        ax=ax
    )
    
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
    
    # 方法2：显示两个类别的SHAP值（可选）
    st.subheader("SHAP Waterfall Plot")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    try:
        # 使用waterfall plot作为替代，通常更稳定
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # 显示正类的SHAP值
            shap_waterfall = shap_values[1][0]  # 正类，第一个样本
            base_val = explainer_shap.expected_value[1] if isinstance(explainer_shap.expected_value, np.ndarray) else explainer_shap.expected_value
        else:
            shap_waterfall = shap_values[0] if isinstance(shap_values, list) else shap_values
            base_val = explainer_shap.expected_value
        
        # 创建Explanation对象
        exp = shap.Explanation(
            values=shap_waterfall,
            base_values=base_val,
            data=input_data.iloc[0],
            feature_names=feature_names
        )
        
        # 绘制waterfall图
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig2)
        
    except Exception as e:
        st.warning(f"Waterfall plot失败: {e}")
        # 回退到summary plot
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        shap.summary_plot(
            shap_values if not isinstance(shap_values, list) else shap_values[1],
            input_data,
            feature_names=feature_names,
            show=False,
            plot_type="bar"
        )
        plt.tight_layout()
        st.pyplot(fig3)
        
except Exception as e:
    st.error(f"SHAP解释失败: {e}")
    st.info("尝试其他可视化方法...")
    
    # 备用方案：显示特征重要性
    st.subheader("特征重要性")
    
    # 获取特征重要性（如果模型支持）
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(range(len(importance_df)), importance_df['重要性'])
        ax_imp.set_yticks(range(len(importance_df)))
        ax_imp.set_yticklabels(importance_df['特征'])
        ax_imp.set_xlabel('特征重要性')
        ax_imp.set_title('模型特征重要性')
        plt.tight_layout()
        st.pyplot(fig_imp)

# LIME Explanation
st.subheader("LIME Explanation")
try:
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
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
    
except Exception as e:
    st.error(f"LIME解释失败: {e}")
    st.info("请确保已安装lime库：pip install lime")
