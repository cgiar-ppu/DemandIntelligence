import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px

st.title("Stakeholder Needs vs Supply Matcher")

# Upload datasets
needs_file = st.file_uploader("Upload Needs Dataset (CSV or Excel)", type=['csv', 'xlsx'])
supply_file = st.file_uploader("Upload Supply Dataset (CSV or Excel)", type=['csv', 'xlsx'])

if needs_file and supply_file:
    # Load data
    if needs_file.name.endswith('.csv'):
        needs_df = pd.read_csv(needs_file)
    else:
        needs_df = pd.read_excel(needs_file)
        
    if supply_file.name.endswith('.csv'):
        supply_df = pd.read_csv(supply_file)
    else:
        supply_df = pd.read_excel(supply_file)
    
    # Select columns
    needs_col = st.selectbox("Select column for Needs", needs_df.columns)
    supply_col = st.selectbox("Select column for Supply", supply_df.columns)
    
    if st.button("Process"):
        # Extract texts
        needs_texts = needs_df[needs_col].dropna().tolist()
        supply_texts = supply_df[supply_col].dropna().tolist()
        
        # Combine with labels
        all_texts = needs_texts + supply_texts
        labels = ['Need'] * len(needs_texts) + ['Supply'] * len(supply_texts)
        
        # Embeddings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(all_texts)
        
        # Clustering setup
        umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom')
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model
        )
        
        topics, probs = topic_model.fit_transform(all_texts, embeddings)
        
        # Visualize in 2D
        umap_2d = UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
        
        df = pd.DataFrame({
            'x': umap_2d[:, 0],
            'y': umap_2d[:, 1],
            'text': all_texts,
            'type': labels,
            'topic': [str(t) for t in topics]  # Convert to str for coloring
        })
        
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='topic', 
            symbol='type', 
            hover_data=['text', 'type'],
            title='Cluster Visualization'
        )
        st.plotly_chart(fig)
        
        # Topic info
        st.subheader("Topic Information")
        st.dataframe(topic_model.get_topic_info())
        
        # Matches per topic
        st.subheader("Matches per Topic")
        matches = {}
        for topic in set(topics):
            if topic != -1:
                topic_needs = [t for t, l, tp in zip(all_texts, labels, topics) if tp == topic and l == 'Need']
                topic_supplies = [t for t, l, tp in zip(all_texts, labels, topics) if tp == topic and l == 'Supply']
                if topic_needs and topic_supplies:
                    matches[topic] = {'needs': topic_needs, 'supplies': topic_supplies}
        
        for topic, data in matches.items():
            st.subheader(f"Topic {topic}")
            st.write("**Needs:**")
            for n in data['needs']:
                st.write(f"- {n}")
            st.write("**Supplies:**")
            for s in data['supplies']:
                st.write(f"- {s}")
