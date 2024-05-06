# CS217-final-project

[project description](https://github.com/Rsirp0c/CS217-final-project/blob/main/src/final-project.pdf)

## How to run the code using virtual environment:
1. `CD` to the project directory
2. Run `python3 -m venv venv` to create a virtual environment
3. Run `source venv/bin/activate` to activate the virtual environment
4. Run `pip install -r requirements.txt` to install the required packages
5. Run `python -m spacy download en_core_web_sm` in the terminal
6. Run `streamlit run app.py` to run the code

## Set up the secret keys for dev mode:
1. `CD` to the project directory
2. `mkdir .streamlit`
3. `touch .streamlit/secrets.toml`
4. Add your secret keys to `secrets.toml`
5. Switch to dev mode

## Users can also set their API keys by submitting the API key form on the sidebar
1. [How to get OpenAI API Keys](https://platform.openai.com/api-keys)
2. [How to get Cohere API keys](https://dashboard.cohere.com/api-keys)
3. [How to get PineCone API Keys](https://www.pinecone.io/?utm_term=pinecone%20db&utm_campaign=Brand+-+US/Canada&utm_source=adwords&utm_medium=ppc&hsa_acc=3111363649&hsa_cam=16223687665&hsa_grp=133738612775&hsa_ad=582256510975&hsa_src=g&hsa_tgt=kwd-1628011569784&hsa_kw=pinecone%20db&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=CjwKCAjw3NyxBhBmEiwAyofDYSeG1vWfF6_vzhmYfdbvuS2VQRAqWSd-BzgeMS0-KaMHzU7rG-Oy4hoCWvIQAvD_BwE)

## Steps to use the chat platform
1. Set up the PineCone Index(Vector Database) by submitting the index creation form on the sidebar.
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot3.png">
2. Upload the pdf file through the file uploader.
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot4.png">
3. Select the desired chunking strategy among character text splitter, recursive character text splitter, and spacy text splitter.
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot5.png">
4. Decide on how much chunks you want to retrieve every time you make a query.
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot6.png">
5. Select the LLM to generate responses 
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot7.png">
6. Switch on some advanced functions based on the provided guideline.
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot8.png">
7. Start chat!
<img width="400"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/screenshot9.png">

## Group member:
- Haoran Cheng
- Xiaoran Liu
- Zhenxu Chen
- Haochen Lin

## Screenshots:
<img width="840"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/Screenshot1.png">
<img width="840"  src="https://github.com/Rsirp0c/CS217-final-project/blob/main/src/Screenshot2.png">

