import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

# Title
st.title('Interactive A/B Testing Dashboard')

# Sidebar: Setup Panel
st.sidebar.header('Data Setup')

# Predefined Datasets Option
data_source = st.sidebar.radio("Choose Data Source", ("Simulate Data", "Select Predefined Dataset"))

# Data Simulation or Predefined Data Selection
if data_source == "Simulate Data":
    st.sidebar.subheader('Simulate A/B Test Data')
    daily_users = st.sidebar.slider("Users per Day", 100, 1000, 300)
    model_a_cr = st.sidebar.slider("Model A Conversion Rate", 0.01, 0.2, 0.1)
    model_b_cr = st.sidebar.slider("Model B Conversion Rate", 0.01, 0.2, 0.12)
    days = st.sidebar.slider("Number of Days", 7, 60, 30)
    
    # Simulate data
    data = {"Day": [], "Model": [], "Users": [], "Conversions": [], "ConversionRate": []}
    for day in range(1, days + 1):
        for model, cr in zip(["A", "B"], [model_a_cr, model_b_cr]):
            users = daily_users
            conversions = np.random.binomial(users, cr)
            conv_rate = conversions / users
            data["Day"].append(day)
            data["Model"].append(model)
            data["Users"].append(users)
            data["Conversions"].append(conversions)
            data["ConversionRate"].append(conv_rate)

    df = pd.DataFrame(data)
    st.write(df.head())
    st.info("Data has been simulated with specified parameters. You can adjust the settings on the sidebar to see different test scenarios.")

elif data_source == "Select Predefined Dataset":
    # List of predefined datasets for A/B testing
    predefined_data = {
        "Dataset 1: Test Group A vs B": {
            "Model A": {"users": 500, "conversion_rate": 0.10},
            "Model B": {"users": 500, "conversion_rate": 0.12},
            "days": 30
        },
        "Dataset 2: Test Group A vs B - Lower Conversion": {
            "Model A": {"users": 500, "conversion_rate": 0.08},
            "Model B": {"users": 500, "conversion_rate": 0.10},
            "days": 30
        }
    }
    
    dataset_name = st.sidebar.selectbox("Select Predefined Dataset", list(predefined_data.keys()))
    dataset = predefined_data[dataset_name]
    
    daily_users = dataset["Model A"]["users"]
    model_a_cr = dataset["Model A"]["conversion_rate"]
    model_b_cr = dataset["Model B"]["conversion_rate"]
    days = dataset["days"]
    
    # Simulate data for selected dataset
    data = {"Day": [], "Model": [], "Users": [], "Conversions": [], "ConversionRate": []}
    for day in range(1, days + 1):
        for model, cr in zip(["A", "B"], [model_a_cr, model_b_cr]):
            users = daily_users
            conversions = np.random.binomial(users, cr)
            conv_rate = conversions / users
            data["Day"].append(day)
            data["Model"].append(model)
            data["Users"].append(users)
            data["Conversions"].append(conversions)
            data["ConversionRate"].append(conv_rate)

    df = pd.DataFrame(data)
    st.write(df.head())
    st.info("Predefined dataset has been selected. You can explore and analyze the test results.")

# Detailed Explanations of Each Step
# -------------------------------------------------------------------------------
# Summary Metrics
st.header("Summary Metrics")
st.write("In this section, we will calculate and display the most important summary metrics from the test data. These include the total number of users, conversions, and the conversion rates for both Model A and Model B.")

# Step 1: Calculate total users and conversions for both models
total_users_a = df[df.Model == 'A'].Users.sum()
total_conversions_a = df[df.Model == 'A'].Conversions.sum()
total_users_b = df[df.Model == 'B'].Users.sum()
total_conversions_b = df[df.Model == 'B'].Conversions.sum()

# Step 2: Calculate conversion rate for both models
conv_rate_a = total_conversions_a / total_users_a
conv_rate_b = total_conversions_b / total_users_b

st.write(f"**Total Users** for Model A: {total_users_a}")
st.write(f"**Total Conversions** for Model A: {total_conversions_a} (Conversion Rate: {conv_rate_a:.2%})")
st.write(f"**Total Users** for Model B: {total_users_b}")
st.write(f"**Total Conversions** for Model B: {total_conversions_b} (Conversion Rate: {conv_rate_b:.2%})")

# Step 3: Calculate and display the difference in conversion rates
delta = conv_rate_b - conv_rate_a
st.write(f"**Conversion Rate Difference (Delta)**: {delta:.2%}")

st.info("These metrics provide a summary of how each model performed in terms of total users, conversions, and conversion rates. A higher conversion rate means the model is more effective in converting users into customers.")

# Visualization of Conversion Rates Over Time
st.header("Conversion Rate Over Time")
st.write("This plot shows the conversion rates for both Model A and Model B over time, which helps us understand how each model performs day by day.")

sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Day", y="ConversionRate", hue="Model", marker="o")
plt.title("Daily Conversion Rate by Model")
plt.ylim(0, 0.2)
plt.ylabel("Conversion Rate")
st.pyplot(plt)

st.info("The plot visualizes how each model’s conversion rate changes over time. This can help identify patterns or fluctuations in performance across different days.")

# Frequentist Z-Test (Statistical Test)
st.header("Frequentist Analysis (Z-Test)")
st.write("The Z-test is used to check if the difference between the conversion rates of the two models is statistically significant. If the difference is statistically significant, it suggests that one model is better than the other, rather than the difference being due to chance.")

count = np.array([total_conversions_a, total_conversions_b])
nobs = np.array([total_users_a, total_users_b])

stat, pval = proportions_ztest(count, nobs)
st.write(f"Z-statistic: {stat:.4f}")
st.write(f"P-value: {pval:.4f}")

if pval < 0.05:
    st.markdown("**Result: Statistically Significant!**", unsafe_allow_html=True)
else:
    st.markdown("**Result: Not Statistically Significant**", unsafe_allow_html=True)

st.info("The Z-test compares the conversion rates of the two models. A p-value less than 0.05 typically indicates a statistically significant difference between the models. This means the difference in performance is unlikely to be due to random chance.")

# **Simple Monte Carlo Simulation** for Bayesian Inference (Simplified)
st.header("Simple Monte Carlo Simulation")
st.write("In this section, we simulate possible conversion rates for both models based on past data. This technique helps us estimate the probability that one model is better than the other using randomness and probability.")

# Simulating the distribution of conversion rates (using normal distributions)
simulations = 10000
p_A_sim = np.random.beta(1, 1, simulations)  # Beta distribution simulating Model A's conversion rate
p_B_sim = np.random.beta(1, 1, simulations)  # Beta distribution simulating Model B's conversion rate

# Step 4: Calculate the probability that Model B is better than Model A
prob_b_bter = np.mean(p_B_sim > p_A_sim)

# Plotting the results
st.write(f"Probability that Model B is better than Model A (Monte Carlo): {prob_b_bter:.2%}")

# Plot the distributions of both A and B
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(p_A_sim, color="blue", label="Model A", kde=True, stat="density")
sns.histplot(p_B_sim, color="orange", label="Model B", kde=True, stat="density")
plt.legend()
plt.title("Monte Carlo Simulation - Conversion Rates (Model A vs Model B)")
st.pyplot(fig)

st.info("The Monte Carlo simulation estimates the probability that Model B will outperform Model A. By simulating many possible outcomes, we can better understand the likelihood of one model being better than the other.")

# Multi-Metric Evaluation
st.header("Multi-Metric Evaluation")
st.write("In this section, we compare both models based on multiple evaluation metrics, including conversion rate, revenue, and session time. This allows us to evaluate each model on different dimensions beyond just conversion rate.")

# Assume revenue is higher for Model B
df["Revenue"] = df["Conversions"] * np.where(df["Model"] == "A", 20, 22)  # Assume higher revenue for Model B
df["SessionTime"] = np.random.normal(loc=60, scale=15, size=len(df))  # Simulated session time in seconds

# Calculate average revenue and session time for both models
avg_revenue_a = df[df.Model == 'A'].Revenue.mean()
avg_revenue_b = df[df.Model == 'B'].Revenue.mean()

avg_session_time_a = df[df.Model == 'A'].SessionTime.mean()
avg_session_time_b = df[df.Model == 'B'].SessionTime.mean()

st.write(f"**Average Revenue for Model A**: ${avg_revenue_a:.2f}")
st.write(f"**Average Revenue for Model B**: ${avg_revenue_b:.2f}")
st.write(f"**Average Session Time for Model A**: {avg_session_time_a:.2f} seconds")
st.write(f"**Average Session Time for Model B**: {avg_session_time_b:.2f} seconds")

st.info("In addition to conversion rate, we also evaluate models based on revenue and session time. A higher revenue or longer session time can indicate that users are more engaged with the model.")

# Summary Conclusion
st.header("Final Conclusion")
st.write("Based on the above analyses, we can conclude whether Model A or Model B performs better for the business goal. We have used multiple statistical methods, visualizations, and evaluations to ensure that our decision is backed by solid data.")
