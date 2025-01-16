import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import os 
from PIL import Image

st.set_page_config(page_title="Analysis of Integrated Reports")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


# Function to display real images
def display_real_images(report):
    # Function to display images on a separate page
    st.title("Real Images")
    real_images = sorted(os.listdir(os.path.join("./Real")))  # Add your image paths or URLs here
    
    image_urls = [os.path.join("./Real", img) for img in real_images if img.find(report.split('.')[0])>=0]

    if len(image_urls) == 0:
        st.write("No real images are found for this report.")
        return
    
    # Display images in a grid
    cols = st.columns(2)  # Adjust the number for the number of columns you want

    # Track the total "height" in each column
    column_heights = [0] * 2

    for img_url in image_urls:
        # Get image height
        with Image.open(img_url) as img:
            img_height = img.size[1]  # Get height of the image

        # Find the column with the smallest current height
        shortest_column = column_heights.index(min(column_heights))

        # Add the image to the chosen column
        with cols[shortest_column]:
            st.image(img_url, use_container_width=True)
        
        # Update the column height
        column_heights[shortest_column] += img_height

# Function to display visual aids
def display_visual_aid(report):
    # Function to display images on a separate page
    st.title("Visual Aids")
    visual_aids = sorted(os.listdir(os.path.join("./Visual Aid")))  # Add your image paths or URLs here
    
    image_urls = [os.path.join("./Visual Aid", img) for img in visual_aids if img.find(report.split('.')[0])>=0]

    if len(image_urls) == 0:
        st.write("No visual aids are found for this report.")
        return

    # Display images in a grid
    cols = st.columns(2)  # Adjust the number for the number of columns you want
    
    # Track the total "height" in each column
    column_heights = [0] * 2

    for img_url in image_urls:
        # Get image height
        with Image.open(img_url) as img:
            img_height = img.size[1]  # Get height of the image

        # Find the column with the smallest current height
        shortest_column = column_heights.index(min(column_heights))

        # Add the image to the chosen column
        with cols[shortest_column]:
            st.image(img_url, use_container_width=True)
        
        # Update the column height
        column_heights[shortest_column] += img_height

@st.cache_data
def load_data():
    summaries = {}
    with open("summary_results.json", "r") as f: 
        summaries = json.load(f)

    topic_results = {}
    with open("topic_results.json", "r") as f: 
        topic_results = json.load(f)

    reports = []
    with open("reports.json", "r") as f: 
        reports = json.load(f)

    results = {}
    with open("results.json", "r") as f: 
        results = json.load(f)

    stats = {}
    with open("stats.json", "r") as f: 
        stats = json.load(f)

    return summaries, topic_results, reports, results, stats

summaries, topic_results, reports, results, stats = load_data()

st.title("Analysis of Integrated Reports Disclosed in Türkiye")


option = st.selectbox(
    "Please choose a report",
    reports.keys()
)


if option == None: 
    st.write("Please choose a report")
else: 
    st.title(option)
    
    report = reports[option]

    # Displaying the Statistics of the Report
    st.subheader("Statistics Per Report")

    page_count = results[report]["Number of pages"]
    real_count = results[report]['Real']
    synthetic_count = results[report]['Synthetic']
    visual_aid_count = results[report]['visual_aid']
    non_visual_count = results[report]['non_visual']

    avg_page_count = stats['avg_page_count']
    avg_real_count_per_page = stats['avg_real_count_per_page']
    avg_synthetic_count_per_page = stats['avg_synthetic_count_per_page']
    avg_visual_aid_count_per_page = stats['avg_visual_aid_count_per_page']
    avg_non_visual_count_per_page = stats['avg_visual_aid_count_per_page']
    avg_real_count = stats['avg_real_count']
    avg_synthetic_count = stats['avg_synthetic_count']
    avg_visual_aid_count = stats['avg_visual_aid_count']
    avg_non_visual_count = stats['avg_visual_aid_count']


    # Data for bar chart
    metrics = ['Page Count', 'Real Images', 'Synthetic Images', 'Visual Aids', 'Non-Aid Visuals']
    report_values = [page_count, results[report]['Real'], results[report]['Synthetic'], results[report]['visual_aid'], results[report]['non_visual']]
    average_values = [avg_page_count, avg_real_count, avg_synthetic_count, avg_visual_aid_count, avg_non_visual_count]

    # Create a DataFrame
    data = pd.DataFrame({
        "Metric": metrics,
        "Report": report_values,
        "Average": average_values
    })

    # Reorder the data so it's in the correct order
    data = data.set_index("Metric")

    # Create a Plotly bar chart
    fig = px.bar(data, x=data.index, y=["Report", "Average"], barmode='group', title="")

    # Layout: Place table and bar chart in the same row
    col1, col2 = st.columns(2)

    # Add table in the first column
    with col1:
        st.write("### Data Table")
        st.dataframe(data)

    # Add bar chart in the second column
    with col2:
        st.write("### Bar Chart")
        st.plotly_chart(fig)


    # Displaying the Per Page Statistics of the Report
    st.subheader("Statistics Per Page")

    # Data for bar chart
    metrics = ['Real Images', 'Synthetic Images', 'Visual Aids', 'Non-Aid Visuals']
    report_values = [results[report]['Real'] / page_count, results[report]['Synthetic'] / page_count, results[report]['visual_aid'] / page_count, results[report]['non_visual'] / page_count]
    average_values = [avg_real_count_per_page, avg_synthetic_count_per_page, avg_visual_aid_count_per_page, avg_non_visual_count_per_page]

    # Create a DataFrame
    data = pd.DataFrame({
        "Metric": metrics,
        "Report": report_values,
        "Average": average_values
    })

    # Reorder the data so it's in the correct order
    data = data.set_index("Metric")

    # Create a Plotly bar chart
    fig = px.bar(data, x=data.index, y=["Report", "Average"], barmode='group', title="")

    # Layout: Place table and bar chart in the same row
    col1, col2 = st.columns(2)

    # Add table in the first column
    with col1:
        st.write("### Data Table")
        st.dataframe(data)

    # Add bar chart in the second column
    with col2:
        st.write("### Bar Chart")
        st.plotly_chart(fig)


    # Create two columns
    col1, col2 = st.columns(2)

    real_clicked = False
    visual_clicked = False
    # Place a button in each column
    with col1:
        if st.button("Display Real Images"):
            real_clicked = not real_clicked
            visual_clicked = False
            # display_real_images(report)

    with col2:
        if st.button("Display Visual Aids"):
            visual_clicked = not visual_clicked
            real_clicked = False
            # display_visual_aid(report)

    if real_clicked:
        display_real_images(report)

    if visual_clicked:
        display_visual_aid(report)


    # Displaying the extracted topics
    st.header("Topics")
    topic_words = []
    topics = topic_results[report]

    for i,topic in enumerate(topics): 
        words = [ word.lower() for word in topic['words'] if len(word.strip())!=0]
        words = ' '.join(set(words))
        st.write(f"**Topic {i+1}:** {words}")


    # Displaying the topic embedding centers on a plane
    reduced_centers = []
    for topic in topics:
        reduced_centers.append(topic['reduced_centers'])

    reduced_centers = np.array(reduced_centers)
    st.subheader("Relationships Between Topics")
    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='blue', marker='o', label="Topic Centers")

    # Annotate the points with topic numbers (or other identifiers)
    for i, center in enumerate(reduced_centers):
        ax.annotate(f"Topic {i+1}", (center[0], center[1]))

    # Set titles and labels
    ax.set_title("Topic Embedding Centers")
    ax.grid(True)

    # Display plot in Streamlit
    st.pyplot(fig)

    # Summary Part
    st.header("Summary")
    summary = summaries[report]

    for part in summary.split('\n'):
        if len(".".join(part.split('.')[:-1]))<10:
            continue
        st.write("\t"+".".join(part.split('.')[:-1])+".\n")
