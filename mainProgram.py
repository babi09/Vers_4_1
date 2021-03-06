# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import modelDeployment
import funcs_ha_use
from PIL import Image
from nibabel import FileHolder, Nifti1Image
from io import BytesIO
from skimage import measure
import numpy as np

# streamlit interface
import plotly.graph_objects as go
st.sidebar.title('Organ Detection and Segmentation')
flag_Liver_Model = 0

# upload file
@st.cache
def loadData(dataAddress):
    img_vol = funcs_ha_use.readVolume4(dataAddress)
    return img_vol

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


#uploaded_nii_file = file_selector()

#st.write('You selected `%s`' % uploaded_nii_file)


uploaded_nii_file = st.sidebar.file_uploader("Select file:", type=['nii'])
# print (uploaded_nii_file)

if uploaded_nii_file is not None:
    rr = uploaded_nii_file.read()
    bb = BytesIO(rr)
    fh = FileHolder(fileobj=bb)
    img = Nifti1Image.from_file_map({'header': fh, 'image': fh})


    #img_vol = Image.open(uploaded_nii_file)
    #content = np.array(img_vol)  # pil to cv
    #print('yes')
    img_vol = loadData(img)
    # plot the data
    # using three column
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # plot the slider
    n_slices1 = img_vol.shape[2]
    slice_i1 = col1.slider('Slice - Axial', 0, n_slices1, int(n_slices1 / 2))

    n_slices2 = img_vol.shape[0]
    slice_i2 = col2.slider('Slice - Coronal', 0, n_slices2, int(n_slices2 / 2))

    n_slices3 = img_vol.shape[1]
    slice_i3 = col3.slider('Slice - Sagittal', 0, n_slices3, int(n_slices3 / 2))

# plot volume
    fig, ax = plt.subplots()
    plt.axis('off')
    def plotImage(img_vol, slice_i):
        selected_slice = img_vol[:, :, slice_i, 1]

        ax.imshow(selected_slice, 'gray', interpolation='none')
        return fig
    fig = plotImage(img_vol, slice_i1)
    #plot = st.pyplot(fig)

    #plot coronal view
    fig1, ax1 = plt.subplots()
    plt.axis('off')
    def plotImageCor(img_vol, slice_i):
        selected_slice2 = img_vol[slice_i, :, :, 1]
        print ('image vol: ')
        print(img_vol.shape)
        print('length:')
        print(len(selected_slice2))
        print('dim 2')
        print(img_vol.shape[2])
        #tr = transforms.Affine2D().rotate_deg(90).translate(len(selected_slice2), 0)

        #ax1.imshow(selected_slice2,'gray', transform=tr + ax1.transData,  interpolation='none')
        rotateIm = list(reversed(list(zip(*selected_slice2))))
        ax1.imshow(rotateIm, 'gray', interpolation='none')
        #print(selected_slice2.shape)
        return fig1

    fig1 = plotImageCor(img_vol, slice_i2)

    # plot sagittal view
    fig2, ax2 = plt.subplots()
    plt.axis('off')

    def plotImageSag(img_vol, slice_i):
        selected_slice3 = img_vol[:, slice_i, :, 1]

        rotateIm = list(reversed(list(zip(*selected_slice3))))
        ax2.imshow(rotateIm, 'gray', interpolation='none')
        # print(selected_slice2.shape)
        return fig2

    fig2 = plotImageSag(img_vol, slice_i3)
# select organ to segment
#     option = st.sidebar.selectbox('Select organ', ('Kidneys', 'Liver', 'Pancreas'))
#     segmentation = st.sidebar.button('Perform Segmentation')
    option = st.sidebar.radio('Select Organ to segment', ['None', 'Kidney', 'Liver', 'Pancreas', 'Psoas'], index=0)

    if option == 'Liver':
        # load segmentation model
        # perform segmentation
        maskSegment, mask = modelDeployment.runDeepSegmentationModel('Liver', img)
        # plot segmentation mask
        fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Liver')
        fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Liver')
        fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Liver')

   
    if option == 'Psoas':
        # load segmentation model
        # perform segmentation
        maskSegment, mask = modelDeployment.runDeepSegmentationModel('Psoas', img)

        # plot segmentation mask
        fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Psoas')
        fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Psoas')
        fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Psoas')


    # plot the three view (axial, sagittal and coronal)



# plot volume
    plot = col1.pyplot(fig)
    plot = col2.pyplot(fig1)
    plot = col3.pyplot(fig2)

    if st.sidebar.button('3D visualisation'):
        print(np.min(mask[:]))
        print(np.max(mask[:]))
        verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.0, allow_degenerate=False)

        # pts = np.loadtxt(
        #     np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
        # x, y, z = pts.T


        fig4 = go.Figure(data=[go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],
                                        opacity=0.6,
                                        autocolorscale=True)])



        #verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.0, allow_degenerate=False)
        #fig4 = go.Figure(data=[go.Scatter3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], mode='markers')])
        col4.plotly_chart(fig4)

        # df = pd.DataFrame({
        #     'x': verts[:, 0],
        #     'y': verts[:, 1],
        #     'z': verts[:, 2],
        #     'normalX': normals[:, 0],
        #     'normalY': normals[:, 1],
        #     'normalZ': normals[:, 2],
        #     'colorR': np.abs(verts[:, 0]),
        #     'colorG': np.abs(verts[:, 1]),
        #     'colorB': np.abs(verts[:, 2]),
        # })
        #
        # st.deck_gl_chart(
        #     layers=[{
        #         'id': 'pointCloud',
        #         'radiusPixels': 1,
        #         'type': 'PointCloudLayer',
        #         'data': df,
        #     }])

        # cloud = pv.PolyData(verts).clean()
        #
        # surf = cloud.delaunay_3d(alpha=5)
        # surf.plot()
       #  shell = surf.extract_geometry().triangulate()
       #  #decimated = shell.decimate(0.4).extract_surface().clean()
       #  #decimated.compute_normals(cell_normals=True, point_normals=False, inplace=True)
       #
       #  #centers = decimated.cell_centers()
       # # centers.translate(decimated['Normals'] * 10.0)
       #
       #  p = pv.Plotter(notebook=False)
       #  p.add_mesh(shell, color="r")
       #  p.link_views()
       #  p.show()

        # Make the xyz points
        # theta = np.linspace(-10 * np.pi, 10 * np.pi, 100)
        # z = np.linspace(-2, 2, 100)
        # r = z ** 2 + 1
        # x = r * np.sin(theta)
        # y = r * np.cos(theta)
        # points = np.column_stack((x, y, z))
        #
        # spline = pv.Spline(points, 500).tube(radius=0.1)
        # spline.plot(scalars='arc_length', show_scalar_bar=False)



