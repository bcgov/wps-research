import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# CSV data as string
csv_data = """Fire Name,Daily FFMC,Daily ISI,Daily FWI,
,,,,
Mt Christie,96.6,15.7,50,
Kelowna,,,,
Rock Creek,98,17.8,57,
Elaho,,,,
Puntzi Lake,,,,
Fort St John,,,,
Elephant Hill,95.6,15.5,49.4,
White Lake (Cariboo) ,95.9,17.2,56,
Hanceville (Cariboo) ,95.1,28.8,76.4,
Plateau (Cariboo) ,,,,
Williams Lake,,,,
100 Mile,,,,
Pentiction (no fire),,,,
Princeton,97.9,23,66,
Burns Lake/Southside,,,,
Telegraph Creek,,,,
Shovel Lake,,,,
Lytton,96.5,18.2,58,
Tremont Ck,,,,
Sparks Lake,97.3,18.5,58,
,98.7,24.2,68,
,98.8,35.1,85,
Battleship Mtn,,,,
"West Kelowna 
(McDougall Ck)",95.2,13.2,47,
"Adams Lake
(Bush Ck East)",94,10.2,39,
"Gunn Lake
(Downton Lk)",94.4,13,43,
St Mary's River,95,38.2,72.5,
Ross Moore Lake,93.6,13.8,34,
Parker Lake,95.4,37.1,70.5,
Slocan Complex,96.7,18.6,36.5,
Wesley Ridge,,,,
Chilcotin Fires,,,,
Fort Nelson,,,,"""

# Read CSV data
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# Remove the trailing comma column if it exists
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove rows with any missing values
df_clean = df.dropna()

# Remove rows where Fire Name is empty
df_clean = df_clean[df_clean['Fire Name'].str.strip() != '']

print("Clean data:")
print(df_clean)
print(f"\nNumber of fires with complete data: {len(df_clean)}")

# Extract data for plotting
fire_names = df_clean['Fire Name'].values
ffmc = df_clean['Daily FFMC'].values
isi = df_clean['Daily ISI'].values
fwi = df_clean['Daily FWI'].values

# Create 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
scatter = ax.scatter(ffmc, isi, fwi, c=fwi, cmap='YlOrRd', s=100, alpha=0.7, edgecolors='black', linewidth=1)

# Add labels for each point
for i, name in enumerate(fire_names):
    ax.text(ffmc[i], isi[i], fwi[i], f'  {name}', fontsize=8, ha='left')

# Set labels and title
ax.set_xlabel('Daily FFMC (Fine Fuel Moisture Code)', fontsize=10, labelpad=10)
ax.set_ylabel('Daily ISI (Initial Spread Index)', fontsize=10, labelpad=10)
ax.set_zlabel('Daily FWI (Fire Weather Index)', fontsize=10, labelpad=10)
ax.set_title('BC Fire Weather Data - 3D Visualization', fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Fire Weather Index (FWI)', fontsize=10)

# Adjust viewing angle for better visibility
ax.view_init(elev=20, azim=45)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plt.savefig('fire_data_3d_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to 'fire_data_3d_plot.png'")

# Show the plot
plt.show()

