import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")  

st.title("Hyperparameter Tuning Results")
intro, settings = st.columns([1, 1.5])
with intro:
    '''
    This is the visualization of the hyperparameter tuning results on `snowflake-arctic-embed-m` model.\n
    While we initially tried to run test on all the embedding models, but later was discouraged due to the time it took to run the tests.
    I runned a grid sreach including the following hyperparameters:
    '''
    with st.expander("Click here to see the hyperparameters"):
        '''
        - Vector calculation method: cosine, euclidean, dotproduct
        - Chunking: character text splitter, recursive character text splitter, spacy text splitter
        - Prompt Extension: True, False
        - Top K: 16(10%), 32(20%), 48(30%)
        - These grid search run on the same pdf file: [the latest KAN neural network essay](https://arxiv.org/pdf/2404.19756) by asking these questions:
            - What is the theoretical basis for Kolmogorov-Arnold Networks, and how do they differ from traditional Multi-Layer Perceptrons (MLPs)?
            - How do KANs address the issue of the curse of dimensionality better than MLPs?
            - What specific applications or types of problems are KANs particularly suited to solve, according to the paper?
            - Does the paper provide evidence of KANs’ interpretability and how it facilitates interaction with human users?
            - What future research directions does the paper suggest for improving or expanding the capabilities of KANs?
        '''
        st.image("src/screenshot/test1.png",caption="Database config options")
        st.image("src/screenshot/test2.png",caption="RAG fine tune options")
        
with settings:
    # Load data
    df = pd.read_csv('hyperparameter_tuning_results_score.csv')

    # User choices for grouping and data comparison
    group, data = st.columns(2)
    with group:
        group_feature = st.multiselect(
            "X axis: Select `group by` feature",
            ["Metric", "Chunking", "Prompt Extension", "Top K", "Prompt"],
            ["Metric", "Chunking"]
        )
    with data:
        data = st.selectbox(
            "Y axis: metric to compare?",
            ("weighted_score", "cosine_similarity")
        )   

    # User choice for y-axis range
    values = st.slider(
        "Select a range of values",
        0.0, 1.0, (0.5, 1.0))
    
    one,two = st.columns([3,1])
    with one:
        viz = st.radio("Select Visualization", ["Bar Chart", "Line Chart","Area Chart"])
    with two:
        st.write("")
        st.write("")
        'click below to go back 👇'
        st.page_link("app.py", label="Main App", icon="🏠")

"---"

table, plot = st.columns([1,1.8])

# df['Combined_Group'] = df[group_feature].apply(lambda x: ' | '.join(x.astype(str)), axis=1)
# grouped = df.groupby('Combined_Group')[data].mean().reset_index()
# grouped_sorted = grouped.sort_values(by=data, ascending=False)

# with table:
#     st.write(grouped_sorted)

# with plot:
#     if viz == "Bar Chart":
#         # Creating a bar chart using Plotly
#         fig = px.bar(df, x= group_feature, y=data,
#                     title='Bar Chart of Data by Group',
#                     labels={'Combined_Group': 'Combined Group', data: data},
#                     color=data)
#     # Setting y-axis limits
#     fig.update_layout(yaxis_range=values)  # Adjust these limits as needed for your data

#     # Displaying the plot in Streamlit
#     st.plotly_chart(fig, use_container_width=True)
    
# Combine and group data
df['Combined_Group'] = df[group_feature].apply(lambda x: ' | '.join(x.astype(str)), axis=1)
grouped = df.groupby('Combined_Group')[data].mean().reset_index()
grouped_sorted = grouped.sort_values(by=data, ascending=False)

table, plot = st.columns([1,1.8])

with table:
    st.write(grouped_sorted)

with plot:
    if viz == "Bar Chart":
        # Creating a bar chart using Plotly
        fig = px.bar(grouped_sorted, x='Combined_Group', y=data,
                    title='Bar Chart of Data by Group',
                    labels={'Combined_Group': 'Combined Group', data: data},
                    color=data)

        # Setting y-axis limits
    elif viz == "Line Chart":
        # Creating a bar chart using Plotly
        fig = px.line(grouped_sorted, x='Combined_Group', y=data,
                    title='Line Chart of Data by Group',
                    labels={'Combined_Group': 'Combined Group', data: data},
                    markers=True)
    elif viz == "Area Chart":
        # Creating a bar chart using Plotly
        fig = px.area(grouped_sorted, x='Combined_Group', y=data,
                    title='Area Chart of Data by Group',
                    labels={'Combined_Group': 'Combined Group', data: data},
                    markers=True)

    # Setting y-axis limits
    fig.update_layout(yaxis_range=values)  # Adjust these limits as needed for your data

    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
