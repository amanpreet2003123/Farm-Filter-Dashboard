import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Farm & Filter Analytics Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #2e7d32;
    }
    h2 {
        color: #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('farm_filter_customer_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'farm_filter_customer_data.csv' is in the same directory.")
        return None

df = load_data()

if df is not None:
    # Header
    st.title("🌿 Farm & Filter Customer Analytics Dashboard")
    st.markdown("### Data-Driven Insights for Organic Cafe & Wellness Hub")
    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Age group filter
    age_groups = ['All'] + list(df['Age_Group'].unique())
    selected_age = st.sidebar.selectbox("Age Group", age_groups)

    # Income filter
    income_levels = ['All'] + list(df['Annual_Income'].unique())
    selected_income = st.sidebar.selectbox("Annual Income", income_levels)

    # Remote work filter
    remote_work_options = ['All'] + list(df['Remote_Work_Frequency'].unique())
    selected_remote = st.sidebar.selectbox("Remote Work Frequency", remote_work_options)

    # Health consciousness filter
    health_range = st.sidebar.slider(
        "Health Consciousness Score",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['Age_Group'] == selected_age]
    if selected_income != 'All':
        filtered_df = filtered_df[filtered_df['Annual_Income'] == selected_income]
    if selected_remote != 'All':
        filtered_df = filtered_df[filtered_df['Remote_Work_Frequency'] == selected_remote]
    filtered_df = filtered_df[
        (filtered_df['Health_Consciousness_Score'] >= health_range[0]) &
        (filtered_df['Health_Consciousness_Score'] <= health_range[1])
    ]

    # Key Metrics
    st.header("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Customers", len(filtered_df))
    with col2:
        avg_spending = filtered_df['Avg_Spending_Per_Visit'].mean()
        st.metric("Avg Spending/Visit", f"${avg_spending:.2f}")
    with col3:
        high_interest = len(filtered_df[filtered_df['Interest_In_FF'] >= 4])
        st.metric("High Interest Customers", high_interest)
    with col4:
        avg_health = filtered_df['Health_Consciousness_Score'].mean()
        st.metric("Avg Health Score", f"{avg_health:.2f}")
    with col5:
        remote_workers = len(filtered_df[filtered_df['Remote_Work_Frequency'].str.contains('Yes', na=False)])
        st.metric("Remote Workers", remote_workers)

    st.markdown("---")

    # Row 1: Customer Demographics
    st.header("👥 Customer Demographics & Spending Patterns")
    col1, col2 = st.columns(2)

    with col1:
        # Age group distribution with spending
        age_spending = filtered_df.groupby('Age_Group')['Avg_Spending_Per_Visit'].mean().reset_index()
        age_spending = age_spending.sort_values('Avg_Spending_Per_Visit', ascending=False)

        fig1 = px.bar(
            age_spending,
            x='Age_Group',
            y='Avg_Spending_Per_Visit',
            title='Average Spending by Age Group',
            labels={'Avg_Spending_Per_Visit': 'Average Spending ($)', 'Age_Group': 'Age Group'},
            color='Avg_Spending_Per_Visit',
            color_continuous_scale='Greens'
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Income vs Dining Frequency Heatmap
        income_dining = filtered_df.groupby(['Annual_Income', 'Dining_Frequency']).size().reset_index(name='Count')

        # Create pivot table for heatmap
        pivot_table = income_dining.pivot(index='Annual_Income', columns='Dining_Frequency', values='Count').fillna(0)

        fig2 = px.imshow(
            pivot_table,
            title='Dining Frequency by Income Level (Heatmap)',
            labels=dict(x="Dining Frequency", y="Annual Income", color="Count"),
            color_continuous_scale='Greens',
            aspect='auto'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2: Health & Organic Validation
    st.header("🥗 Health Consciousness & Organic Preferences")
    col1, col2 = st.columns(2)

    with col1:
        # Health consciousness vs spending
        health_spending = filtered_df.groupby('Importance_Organic').agg({
            'Avg_Spending_Per_Visit': ['mean', 'median', 'std']
        }).reset_index()
        health_spending.columns = ['Importance_Organic', 'Mean', 'Median', 'Std']

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=health_spending['Importance_Organic'],
            y=health_spending['Mean'],
            mode='lines+markers',
            name='Mean Spending',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        fig3.update_layout(
            title='Average Spending by Importance of Organic Food',
            xaxis_title='Importance of Organic (1-5)',
            yaxis_title='Average Spending ($)',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Menu preferences
        menu_prefs = {
            'Organic Coffee': filtered_df['Prefers_Organic_Coffee'].sum(),
            'Cold Brew': filtered_df['Prefers_Cold_Brew'].sum(),
            'Fresh Juices': filtered_df['Prefers_Juices'].sum(),
            'Smoothies': filtered_df['Prefers_Smoothies'].sum()
        }

        menu_df = pd.DataFrame(list(menu_prefs.items()), columns=['Item', 'Count'])
        menu_df['Percentage'] = (menu_df['Count'] / len(filtered_df) * 100).round(1)

        fig4 = px.bar(
            menu_df,
            x='Item',
            y='Count',
            title='Top Menu Item Preferences',
            labels={'Count': 'Number of Customers', 'Item': 'Menu Item'},
            color='Count',
            color_continuous_scale='Greens',
            text='Percentage'
        )
        fig4.update_traces(texttemplate='%{text}%', textposition='outside')
        fig4.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Row 3: Workspace & Remote Work Analysis
    st.header("💻 Workspace & Remote Work Insights")
    col1, col2 = st.columns(2)

    with col1:
        # Remote work frequency distribution
        remote_dist = filtered_df['Remote_Work_Frequency'].value_counts().reset_index()
        remote_dist.columns = ['Remote_Work_Frequency', 'Count']

        fig5 = px.pie(
            remote_dist,
            values='Count',
            names='Remote_Work_Frequency',
            title='Remote Work Frequency Distribution',
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig5.update_traces(textposition='inside', textinfo='percent+label')
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        # Workspace need vs spending
        workspace_spending = filtered_df.groupby('Workspace_Need_Score')['Avg_Spending_Per_Visit'].mean().reset_index()

        fig6 = px.line(
            workspace_spending,
            x='Workspace_Need_Score',
            y='Avg_Spending_Per_Visit',
            title='Workspace Need vs Average Spending',
            labels={'Workspace_Need_Score': 'Workspace Need Score (1-5)', 'Avg_Spending_Per_Visit': 'Avg Spending ($)'},
            markers=True
        )
        fig6.update_traces(line_color='green', marker=dict(size=12, color='darkgreen'))
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # Row 4: Customer Priorities & Location Analysis
    st.header("🎯 Customer Priorities & Location Strategy")
    col1, col2 = st.columns(2)

    with col1:
        # Customer priorities ranking
        priorities = {
            'Taste/Quality': filtered_df['Priority_Taste_Quality'].mean(),
            'Price/Value': filtered_df['Priority_Price_Value'].mean(),
            'Location/Convenience': filtered_df['Priority_Location'].mean(),
            'Ambiance': filtered_df['Priority_Ambiance'].mean()
        }

        priorities_df = pd.DataFrame(list(priorities.items()), columns=['Priority', 'Score'])
        priorities_df = priorities_df.sort_values('Score', ascending=True)

        fig7 = px.bar(
            priorities_df,
            x='Score',
            y='Priority',
            orientation='h',
            title='Customer Priority Rankings (Average Scores)',
            labels={'Score': 'Average Score (out of 5)', 'Priority': 'Priority Factor'},
            color='Score',
            color_continuous_scale='Greens'
        )
        fig7.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        # Distance vs Spending and Frequency
        distance_analysis = filtered_df.groupby('Distance_From_Location').agg({
            'Avg_Spending_Per_Visit': 'mean',
            'Customer_ID': 'count'
        }).reset_index()
        distance_analysis.columns = ['Distance', 'Avg_Spending', 'Customer_Count']

        fig8 = go.Figure()
        fig8.add_trace(go.Bar(
            x=distance_analysis['Distance'],
            y=distance_analysis['Avg_Spending'],
            name='Avg Spending',
            marker_color='lightgreen',
            yaxis='y'
        ))
        fig8.add_trace(go.Scatter(
            x=distance_analysis['Distance'],
            y=distance_analysis['Customer_Count'],
            name='Customer Count',
            marker_color='darkgreen',
            yaxis='y2',
            mode='lines+markers'
        ))

        fig8.update_layout(
            title='Location Analysis: Distance vs Spending & Frequency',
            xaxis_title='Distance from Location',
            yaxis=dict(title='Avg Spending ($)', side='left'),
            yaxis2=dict(title='Customer Count', overlaying='y', side='right'),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

    # Row 5: 3D Golden Segment Visualization
    st.header("✨ Golden Segment Analysis")

    # Create interest level categories
    filtered_df['Interest_Level'] = pd.cut(
        filtered_df['Interest_In_FF'],
        bins=[0, 2, 3, 4, 5],
        labels=['Low', 'Moderate', 'High', 'Extremely High']
    )

    fig9 = px.scatter_3d(
        filtered_df,
        x='Health_Consciousness_Score',
        y='Workspace_Need_Score',
        z='Avg_Spending_Per_Visit',
        color='Interest_Level',
        title='3D Golden Segment: Health × Workspace × Spending × Interest',
        labels={
            'Health_Consciousness_Score': 'Health Consciousness',
            'Workspace_Need_Score': 'Workspace Need',
            'Avg_Spending_Per_Visit': 'Avg Spending ($)',
            'Interest_Level': 'Interest in Farm & Filter'
        },
        color_discrete_map={
            'Low': 'lightgray',
            'Moderate': 'yellow',
            'High': 'orange',
            'Extremely High': 'green'
        },
        height=600
    )
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("---")

    # Data Table
    st.header("📋 Filtered Customer Data")
    st.dataframe(
        filtered_df.drop(['Customer_ID'], axis=1),
        use_container_width=True,
        height=400
    )

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="farm_filter_filtered_data.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Farm & Filter Analytics Dashboard | Built with Streamlit | Data Analytics Project</p>
    </div>
    """, unsafe_allow_html=True)
