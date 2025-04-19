import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset
df = pd.read_csv("Results_21Mar2022.csv")

# 2. Select relevant environmental impact variables
env_columns = [
    "mean_ghgs", "mean_land", "mean_watscar", "mean_eut",
    "mean_ghgs_ch4", "mean_ghgs_n2o", "mean_bio", "mean_watuse", "mean_acid"
]

# 3. Group by diet group and calculate the mean for each variable
diet_group_means = df.groupby("diet_group")[env_columns].mean().reset_index()

# 4. Normalize each variable (Min-Max Scaling)
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(diet_group_means[env_columns])
scaled_df = pd.DataFrame(scaled_values, columns=env_columns)
scaled_df["diet_group"] = diet_group_means["diet_group"]

# 5. Map diet group to numerical codes for coloring
diet_mapping = {name: idx for idx, name in enumerate(scaled_df["diet_group"].unique())}
scaled_df["diet_code"] = scaled_df["diet_group"].map(diet_mapping)
diet_legend = {v: k for k, v in diet_mapping.items()}

# 6. Define user-friendly axis labels
labels = {
    "mean_ghgs": "GHG\nEmissions\n(kg)",
    "mean_land": "Land\nUse\n(m²)",
    "mean_watscar": "Water\nScarcity",
    "mean_eut": "Eutrophication\n(g PO₄e)",
    "mean_ghgs_ch4": "CH₄\nEmissions\n(kg)",
    "mean_ghgs_n2o": "N₂O\nEmissions\n(kg)",
    "mean_bio": "Biodiversity\nLoss",
    "mean_watuse": "Water\nUse\n(m³)",
    "mean_acid": "Acidification\nPotential"
}


# 7. Create parallel coordinates plot
fig = px.parallel_coordinates(
    scaled_df,
    color="diet_code",
    dimensions=env_columns,
    color_continuous_scale=px.colors.diverging.Tealrose,
    labels=labels
)

# 8. Enhance layout and color legend
fig.update_layout(
    height=600,
    margin=dict(t=100),
    title=dict(
        text="Environmental Impact Comparison Across Diet Groups (Parallel Coordinates Plot)",
        x=0.5,
        y=0.95,
        font=dict(size=18)
    ),
    coloraxis_colorbar=dict(
        title="Diet Group",
        tickvals=list(diet_legend.keys()),
        ticktext=[diet_legend[k] for k in diet_legend.keys()]
    )
)


# 9. Show the plot in browser
fig.show()

# 10. Save the plot as PNG image (optional for coursework PDF submission)
fig.write_image("diet_environment_parallel_coordinates.png", scale=2)
