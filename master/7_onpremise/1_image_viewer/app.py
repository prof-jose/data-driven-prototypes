import streamlit as st
import pandas as pd


@st.cache_data
def init():
    """Load model and labels."""
    labels_df = pd.read_csv("index.csv")
    return labels_df


labels_df = init()

st.logo("cloud.png")
st.title("Photo search")

n_photos = len(labels_df.file.unique())
st.text(f"Searching in {n_photos} photos")

# Widget for search box with autocomplete:
options = st.multiselect(
    "Search for image tags",
    labels_df.word.unique(),
    placeholder="🔎 Type words"
)

matches = labels_df[labels_df.word.isin(options)]

# Display the number of matches:
if len(matches) > 0:
    if len(matches) == 1:
        st.sidebar.subheader("Showing 1 match")
    else:
        st.sidebar.subheader(f"Showing {len(matches)} matches")

# Display how many matches per word with progress bars:
word_counts = matches.word.value_counts()
for word, count in word_counts.items():
    sidecols = st.sidebar.columns(3)
    with sidecols[0]:
        st.text(f"{word}")
    with sidecols[1]:
        st.progress(count / len(matches))
    with sidecols[2]:
        st.text(f"{count}")


# Display images in a 4x? grid:

num = 0
cols = st.columns(4)

for i, row in matches.iterrows():
    # Add expander with the file name
    with cols[num % 4]:
        st.text(f"{row.word}")
        st.image(row.file)
    num += 1
