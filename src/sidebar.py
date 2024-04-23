import streamlit as st
from pinecone import Pinecone, ServerlessSpec


def add_dataset():
    with st.form('New Dataset', clear_on_submit=True):
        name = st.text_input('Dataset Name',placeholder='name of dataset')
        description = st.text_input('Description', placeholder='Description of dataset')
        model = st.selectbox('Select Embedding model', ['model 1', 'model 2', 'model 3'], help='Different embedding models vectorize text differently')
        metrics = st.selectbox('Select Metrics', ['cosine', 'euclidean', 'dot product'], help='Metrics are used to calculate the similarity between vectors')
        New_dataset = st.form_submit_button('Create New Dataset')
        if New_dataset:
            st.session_state.datasets[name] = [description, model, metrics]
            pc = Pinecone(api_key=st.session_state.api_keys['pinecone_api_key'])
            index_name = name
            pc.create_index(
                name=index_name,
                dimension=8,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            ) 
            index = pc.Index(index_name)
            st.rerun()
            


def add_keys():
    with st.form('API Keys', clear_on_submit=True):
        pinecone_api_key = st.text_input('Pinecone API Key', type="password")
        cohere_api_key = st.text_input('Cohere API Key (optional)', type="password")
        # pinecone_index_name = st.text_input('Pinecone Index Name',type="password" )
        # pinecone_environment = st.text_input('Pinecone Environment', type="password")
        New_keys = st.form_submit_button('Add API Keys')
        if New_keys:
            st.session_state.api_keys['pinecone_api_key'] = pinecone_api_key
            st.session_state.api_keys['cohere_api_key'] = cohere_api_key
            # st.session_state.api_keys['pinecone_index_name'] = pinecone_index_name
            # st.session_state.api_keys['pinecone_environment'] = pinecone_environment

def init_dev_or_prod():
    '''
    This function is used to toggle between development and production environments.

    - session_state is used to store the environment status and api keys throughout a session.
      https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    - secrets are used to store sensitive information such as API keys that are not exposed to the user.
      https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
    '''
    
    if 'environment_status' not in st.session_state:
        st.session_state.environment_status = None

    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'pinecone_api_key': None,
            'cohere_api_key': None,
            # 'pinecone_index_name': None,
            # 'pinecone_environment': None
        }

    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
        # {'dataset1': ['description', 'model1', 'cosine'], ...}

    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None

    with st.sidebar:
        on = st.toggle('Development Environment', False)
        if on:
            st.session_state.environment_status = 'dev'
        else:
            st.session_state.environment_status = 'prod'

        if st.session_state.environment_status == 'dev':
            st.session_state.api_keys['pinecone_api_key'] = st.secrets["PINECONE_API_KEY"]
            st.session_state.api_keys['cohere_api_key'] = st.secrets["COHERE_API_KEY"]
            # st.session_state.api_keys['pinecone_index_name'] = st.secrets['PINECONE_INDEX_NAME']
            # st.session_state.api_keys['pinecone_environment'] = st.secrets['PINECONE_ENVIRONMENT']
            # progress_spin()
            st.success('Development environment is on', icon="✅")
        else:
            if st.session_state.api_keys['pinecone_api_key'] and st.session_state.api_keys['cohere_api_key']:
                with st.expander("edit API keys"):
                    add_keys()
            else:
                add_keys()


def sidebar_func():
    st.sidebar.title("App Settings")
    init_dev_or_prod()
    
    if st.session_state.api_keys['pinecone_api_key'] == None:
        st.sidebar.warning('Please add Pinecone API Key')

    else:
        dataset = st.sidebar.selectbox("Select a dataset", st.session_state.datasets,index=0)
        if dataset:
            st.session_state.current_dataset = dataset
        
        with st.sidebar:
            if st.session_state.datasets == {}:
                    add_dataset()
            else:
                with st.expander("Add New Dataset"):
                    add_dataset() 