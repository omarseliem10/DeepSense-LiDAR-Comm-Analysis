#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import scipy.io as scipyio
import matplotlib.pyplot as plt

from tqdm import tqdm

# Absolute path of the folder containing the units' folders and scenario32.csv
scenario_folder = r"D:\Master\Selected Topics - Communication Technology\LiDAR Digital Twin and Integrated Sensing and Communications\scenario32_2"

# Fetch scenario CSV
try:
    csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
    csv_path = os.path.join(scenario_folder, csv_file)
except:
    raise Exception(f'No csv file inside {scenario_folder}.')

# Load CSV to dataframe
dataframe = pd.read_csv(csv_path)
print(f'Columns: {dataframe.columns.values}')
print(f'Number of Rows: {dataframe.shape[0]}')

#%%
N_BEAMS = 64
n_samples = 300
pwr_rel_paths = dataframe['unit1_pwr_60ghz'].values
pwrs_array = np.zeros((n_samples, N_BEAMS))

for sample_idx in tqdm(range(n_samples)):
    pwr_abs_path = os.path.join(scenario_folder,
                                pwr_rel_paths[sample_idx])
    pwrs_array[sample_idx] = np.loadtxt(pwr_abs_path)
#%%
# Select specific samples to display
selected_samples = [59,70,71]
beam_idxs = np.arange(N_BEAMS) + 1
plt.figure(figsize=(10,6))
plt.plot(beam_idxs, pwrs_array[selected_samples].T)
plt.legend([f'Sample {i}' for i in selected_samples])
plt.xlabel('Beam indices')
plt.ylabel('Power')
plt.grid()
#%%
img_rel_paths = dataframe['unit1_rgb'].values
fig, axs = plt.subplots(figsize=(10,4), ncols=len(selected_samples), tight_layout=True)
for i, sample_idx in enumerate(selected_samples):
    img_path = os.path.join(scenario_folder, img_rel_paths[sample_idx])
    img = plt.imread(img_path)
    axs[i].imshow(img)
    axs[i].set_title(f'Sample {sample_idx}')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    #%%
    # BS position (take the first because it is static)
bs_pos_rel_path = dataframe['unit1_loc'].values[0]
bs_pos = np.loadtxt(os.path.join(scenario_folder,
                                 bs_pos_rel_path))
# UE positions
pos_rel_paths = dataframe['unit2_loc'].values
pos_array = np.zeros((n_samples, 2)) # 2 = lat & lon

# Load each individual txt file
for sample_idx in range(n_samples):
        pos_abs_path = os.path.join(scenario_folder,
                                pos_rel_paths[sample_idx])
        pos_array[sample_idx] = np.loadtxt(pos_abs_path)

# Prepare plot: We plot on top of a Google Earth screenshot
gps_img = plt.imread(scenario_folder+'\Resource\Scenario32-example-sat-img.png')

# Function to transform coordinates
def deg_to_dec(d, m, s, direction='N'):
     if direction in ['N', 'E']:
        mult = 1
     elif direction in ['S', 'W']:
        mult = -1
     else:
        raise Exception('Invalid direction.')

     return mult * (d + m/60 + s/3600)

# GPS coordinates from the bottom left and top right coordinates of the screenshot
gps_bottom_left, gps_top_right = ((deg_to_dec(33, 25, 25.45, 'N'),
                                   deg_to_dec(111, 56, 7.34, 'W')),
                                  (deg_to_dec(33, 25, 27.86, 'N'),
                                   deg_to_dec(111, 56, 5.32, 'W'))) 
#%%
# Important: screenshots taken with orientation towards North
best_beams = np.argmax(pwrs_array, axis=1)
fig, ax = plt.subplots(figsize=(6,8), dpi=100)
ax.imshow(gps_img, aspect='auto', zorder=0,
          extent=[gps_bottom_left[1], gps_top_right[1],
                  gps_bottom_left[0], gps_top_right[0]])

scat = ax.scatter(pos_array[selected_samples,1], pos_array[selected_samples,0], edgecolor='black', lw=0.7,
                  c=(best_beams[selected_samples] / N_BEAMS), vmin=0, vmax=1,
                  cmap=matplotlib.colormaps['jet'])

cbar = plt.colorbar(scat)
cbar.set_ticks([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
cbar.ax.set_yticklabels(['1', '8', '16', '24', '32', '40', '48', '56', '64'])
cbar.ax.set_ylabel('Best Beam Index', rotation=-90, labelpad=10)
ax.scatter(bs_pos[1], bs_pos[0], s=100, marker='X', color='red', label='BS')
ax.legend()
ax.ticklabel_format(useOffset=False, style='plain')
ax.tick_params(axis='x', labelrotation=45)
ax.set_xlabel('Longitude [º]')
ax.set_ylabel('Latitude [º]')

# We see about 2.5 car passes.

#%%
import open3d as o3d
import os
import time

# Path to the folder containing .ply files
folder_path = r"D:\Master\Selected Topics - Communication Technology\LiDAR Digital Twin and Integrated Sensing and Communications\scenario32_2\unit1\lidar_data"

# Get all .ply files in the folder and sort them
ply_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Point Cloud Animation", width=1280, height=720)

# Set background color to black
render_option = vis.get_render_option()
render_option.background_color = [0, 0, 0]  # RGB for black

# Adjust point size for better visibility
render_option.point_size = 3.0  # Increase point size as needed

# Initialize geometry (to reuse it in the loop)
geometry = None

# Add a delay to ensure window is properly created
vis.poll_events()
vis.update_renderer()

for file in ply_files:
    # Load the current .ply file
    pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file))

    # Downsample the point cloud using voxel grid filter
    voxel_size = 0.1
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # Remove noise using statistical outlier removal
    cl, ind = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered = pcd_downsampled.select_by_index(ind)

    if geometry is None:
        # Add the first point cloud to the visualizer
        geometry = pcd_filtered
        vis.add_geometry(geometry)
    else:
        # Update the geometry for subsequent frames
        geometry.points = pcd_filtered.points
        geometry.colors = pcd_filtered.colors
        vis.update_geometry(geometry)

    # Poll events and render the frame
    vis.poll_events()
    vis.update_renderer()

    # Adjust frame delay (e.g., 0.1 seconds per frame for 10 FPS)
    time.sleep(0.01)

# Close the visualization window after the loop
vis.destroy_window()

# %%
