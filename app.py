import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import io

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
        # Build union of original rows used for clustering
        needs_subset = needs_df[needs_df[needs_col].notna()].copy()
        needs_subset["text"] = needs_subset[needs_col].astype(str)
        needs_subset["type"] = "Need"
        supply_subset = supply_df[supply_df[supply_col].notna()].copy()
        supply_subset["text"] = supply_subset[supply_col].astype(str)
        supply_subset["type"] = "Supply"
        combined_subset = pd.concat([needs_subset, supply_subset], ignore_index=True, sort=False)

        # Combine with labels for modeling/visualization
        all_texts = combined_subset["text"].tolist()
        labels = combined_subset["type"].tolist()
        
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
        # Persist union of inputs with assigned topics for download
        combined_subset["topic"] = topics
        st.session_state["union_with_topics"] = combined_subset

    # Render download button if results are available
    if "union_with_topics" in st.session_state:
        output_df = st.session_state["union_with_topics"]
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            output_df.to_excel(writer, index=False, sheet_name="union_with_topics")
        excel_buffer.seek(0)
        st.download_button(
            label="Download clustering union with topics (Excel)",
            data=excel_buffer.getvalue(),
            file_name="clustering_union_with_topics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
