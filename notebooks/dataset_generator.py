#!/usr/bin/env python
# coding: utf-8

# # Dataset Generator:
# - Read low level info from slide.
# - Extract coordinates from XML.
# - Generate patches & masks: ROI-guided + corner augmentation
# - Color normalization: macenko-LS & vahadane-LS.
# - Background extraction.
# - Save CSV. 

# In[31]:


from openslide import OpenSlide, lowlevel 
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw
import math
from matplotlib import cm
import staintools
import glob
import os
from tqdm import tqdm
import cv2 as cv

import xml.etree.ElementTree as ET

Image.MAX_IMAGE_PIXELS = 325755924


# ## Reading low level and the region using OpenSlide

# In[32]:


# Define the path for the slides & labels
slides_path = '/Users/gabriel.jimenez/Documents/project/AT8Dataset/AT8_wsi/'
labels_path = '/Users/gabriel.jimenez/Documents/project/AT8Dataset/AT8_XML_annotations/'

######################################################
# TODO: glob from folder to read all slides and xmls #
######################################################
slides = sorted(glob.glob(os.path.join(slides_path,'*.ndpi')))
annotations = sorted(glob.glob(os.path.join(labels_path,'*.xml')))

# idx = 3
remove_patches = False # for now it is used for WSI 11
for idx in [5, 6, 7, 8, 9, 10, 12, 13, 14]:
    print(idx)
    slide_name = slides[idx]
    label_name = annotations[idx]

    print(slide_name, label_name)

    # Opening the slide image
    slide = lowlevel.open(slide_name)
    keys = lowlevel.get_property_names(slide)
    val = lowlevel.get_property_value(slide,keys[-1])

    # This are important values for nm -> pixel conversion
    offsetX = int(lowlevel.get_property_value(slide, 'hamamatsu.XOffsetFromSlideCentre')) # THIS IS IN NANOMETERS!
    offsetY = int(lowlevel.get_property_value(slide, 'hamamatsu.YOffsetFromSlideCentre')) # THIS IS IN NANOMETERS!

    resX = float(lowlevel.get_property_value(slide, 'openslide.mpp-x')) # THIS IS IN MICRONS/PIXEL FOR LVL 0!
    resY = float(lowlevel.get_property_value(slide, 'openslide.mpp-y')) # THIS IS IN MICRONS/PIXEL FOR LVL 0!

    #######################################################################
    # TODO: check if all dimensions are in nm (ISSUE: different scanners) #
    #######################################################################


    # In[33]:


    slide = OpenSlide(slide_name)

    # Getting slide level dimentions
    slide_levels = slide.level_dimensions

    # Printing important information about the current slide
    print("[INFO] The slide have ", len(slide_levels), " magnification levels:")
    for i in range(len(slide_levels)):
        print(f"   Level {i} (mag x{40/2**i}) with dimensions (in pixels) : {slide_levels[i]}.")

    # Chosing the magnification level
    slide_dim_lvl = 2

    '''
    TODO: put a flag because this is just for debugging. We only need to load the ROI to RAM #
    '''
    # Getting the thumbnail for slide
    # thm = slide.read_region((0, 0), slide_dim_lvl, slide_levels[slide_dim_lvl])


    # ## Extracting coordinates from XML and conversion to pixels

    # In[34]:


    ''' 
    From nano/micro meters to pixel language :) 
    Larger value is X axis (-->)
    '''
    dimsSlide = np.array(slide_levels[0])*[resX,resY] # this is in micrometers :)
    centerPx_X, centerPx_Y = np.array(slide_levels[0])/2
    _, factor = np.array(slide_levels[0])/np.array(slide_levels[slide_dim_lvl])

    sizeX, sizeY = np.array(slide_levels[0])/factor

    # Loading the slide annotations from XML file
    tree = ET.parse(label_name)
    root = tree.getroot()

    # Preparing the annotation container
    labels = []

    # Getting the annotaiton coordinates from the XML file
    for boxes in root:
        for obejcts in boxes:
            type_object = int(obejcts.attrib['Type'])
            for vertices in obejcts:
                temp_obj = []
                for vertex in vertices:
                    y_mm = float(vertex.attrib['Y']) # this is in milimeters!
                    x_mm = float(vertex.attrib['X']) # this is in milimeters!
                    y_p_offset = (y_mm)*1000 - (abs(offsetY)/1000) # this is in micrometers!
                    x_p_offset = (x_mm)*1000 - (abs(offsetX)/1000) # this is in micrometers!
                    y_newCenter = y_p_offset + int(centerPx_Y)*resY # this is in micrometers!
                    x_newCenter = x_p_offset + int(centerPx_X)*resX # this is in micrometers!
                    y = (y_newCenter/resY)/factor # pixels
                    x = (x_newCenter/resX)/factor # pixels

                    ''' Flip '''
                    y = abs(sizeY - y)
                    #x = sizeX - x

                    temp_obj.append([round(x), round(y)])
                labels.append([type_object, np.array(temp_obj)])


    # In[35]:


    '''
    TODO: put a flag because this is just for debugging. We only need to load the ROI to RAM #

    '''
    # Overlaying the first object (Gray matter annotation)
    # print("\n[INFO] The whole-slide image with the gray matter annotation ( dim level", slide_dim_lvl, ")")
    # plt.figure(figsize = (10,10))
    # plt.imshow(thm)
    # plt.plot(labels[0][1][:, 0], labels[0][1][:, 1])
    # for obj, coordinates in labels:
    #     plt.plot(coordinates[:, 0], coordinates[:, 1])
    # plt.show()


    # ## Generate patches & masks: ROI-guided.

    # In[36]:


    # Creating a polygone
    # polygon_patch = Polygon(labels[0][1])

    # Getting the bounding box of the label
    # coords_region = list(polygon_patch.bounds)

    # Getting the size of the label
    # size = (int(coords_region[2]-coords_region[0])+1, int(coords_region[3]-coords_region[1])+1)

    # Extracting the region of the label from the whole-slide-image
    # region = slide.read_region((int(coords_region[0]*factor), int(coords_region[1]*factor)), slide_dim_lvl, (size[0], size[1]))


    # In[37]:


    # Creating mask for the WSI
    mask_ROI = Image.new('L', (int(sizeX), int(sizeY)), 0)
    mask_obj = Image.new('L', (int(sizeX), int(sizeY)), 0)

    coords_list = []
    coords_region_lst = []
    for obj, coordinates in labels:
        if obj == 1: # this are the annotations by Lev.
            coordinates2list = coordinates.tolist()
            tuples = [tuple(x) for x in coordinates2list]
            ImageDraw.Draw(mask_ROI).polygon(tuples, outline=1, fill=1)

            polygon_patch = Polygon(coordinates)
            coords_region_lst.append(list(polygon_patch.bounds))
            # size = (int(coords_region[2]-coords_region[0])+1, int(coords_region[3]-coords_region[1])+1)
    
        if obj == 2:
            coordinates2list = coordinates.tolist()
            tuples = [tuple(x) for x in coordinates2list]
            ImageDraw.Draw(mask_obj).polygon(tuples, outline=1, fill=1)
            
            polygon_obj = Polygon(coordinates)
            coords_obj = list(polygon_obj.bounds)
            coords_list.append(coords_obj)

    # Get the coordinates from the ROI --> this is because not all labels follow the same pattern.
    # min_val = coords_region_lst[0][0]
    # coords_region = coords_region_lst[0]
    # for c in coords_region_lst:
    #     if c[0] < min_val:
    #         min_val = c[0]
    #         coords_region = c    
    # size = (int(coords_region[2]-coords_region[0])+1, int(coords_region[3]-coords_region[1])+1)

    coords_region = []
    size = []
    for c in coords_region_lst:
        coords_region.append(c) 
        size.append((int(c[2]-c[0])+1, int(c[3]-c[1])+1))
    print(coords_region)
    print(size)

    #mask_ROI_WSI = mask_ROI.crop((coords_region[0],coords_region[1],coords_region[0]+size[0],coords_region[1]+size[1]))
    #mask_obj_WSI = mask_obj.crop((coords_region[0],coords_region[1],coords_region[0]+size[0],coords_region[1]+size[1]))

    mask_ROI_WSI = mask_ROI.point(lambda i: i * 255)
    mask_obj_WSI = mask_obj.point(lambda i: i * 255)

    '''
    # Save and plot just to check if it is working properly
    mask_ROI_WSI.point(lambda i: i * 255).save('./name.png')
    mask_obj_WSI.point(lambda i: i * 255).save('./name2.png')
    '''
    # plt.figure()
    # plt.imshow(mask_ROI_WSI, cmap = "gray")
    # #plt.xlim((coords_region[0], coords_region[2]))
    # #plt.ylim((coords_region[3], coords_region[1]))
    # plt.show()

    # plt.figure()
    # plt.imshow(mask_obj_WSI, cmap = "gray")
    # #plt.xlim((coords_region[0], coords_region[2]))
    # #plt.ylim((coords_region[3], coords_region[1]))
    # plt.show()


    # In[38]:


    '''
    This code creates the region and its corresponding patches for the 4 corners of data augmentation.
    '''
    patchSize = [512, 512] # [cols, rows]
    save_path = os.path.join("/Users/gabriel.jimenez/Documents/project/togitlab/dataset/512x512", slide_name.split('/')[-1].split('.ndpi')[0])
    os.makedirs(save_path, exist_ok=True)

    os.makedirs(os.path.join(save_path, "patches"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)

    # mask_WSI: has all the masks for the annotations and is the same size as the WSI.
    # slide: is the original WSI in the level selected. It is not loaded into RAM yet.
    # labels: annotations from XML.
    # coords_list: list of coordinates of annotated objects.

    k = 0
    for coords in coords_list: 
        coords[0] = int(math.floor(coords[0]))
        coords[1] = int(math.floor(coords[1]))
        coords[2] = int(math.ceil(coords[2]))
        coords[3] = int(math.ceil(coords[3]))

        size_obj = [coords[2]-coords[0]+1, coords[3]-coords[1]+1]
        new_region_size = np.array(patchSize) - np.array(size_obj)        

        new_coords = (coords[0]-new_region_size[0]/2, coords[1]-new_region_size[1]/2, coords[2]+new_region_size[0]/2, coords[3]+new_region_size[1]/2)        
        new_region_obj = slide.read_region((int(new_coords[0]*factor), int(new_coords[1]*factor)), slide_dim_lvl, (patchSize[0], patchSize[1]))
        new_mask_obj = mask_obj_WSI.crop((int(new_coords[0]),int(new_coords[1]),int(new_coords[0]+patchSize[0]),int(new_coords[1]+patchSize[1])))

        new_region_obj.save(os.path.join(save_path,f'patches/patch_{k:04}.png'))
        new_mask_obj.save(os.path.join(save_path,f'masks/mask_{k:04}.png'))

        '''
        Create 4 additional samples per object/patch ... the annotated object will be in each corner
        '''
        new_region_corner1 = slide.read_region((int(coords[0]*factor), int(coords[1]*factor)), slide_dim_lvl, (patchSize[0], patchSize[1]))
        new_mask_corner1 = mask_obj_WSI.crop((coords[0],coords[1],coords[0]+patchSize[0],coords[1]+patchSize[1]))

        new_region_corner1.save(os.path.join(save_path,f'patches/patch_{k:04}_c1.png'))
        new_mask_corner1.save(os.path.join(save_path,f'masks/mask_{k:04}_c1.png'))

        corner2_coords = (coords[2] - patchSize[0], coords[1])
        new_region_corner2 = slide.read_region((int(corner2_coords[0]*factor), int(corner2_coords[1]*factor)), slide_dim_lvl, (patchSize[0], patchSize[1]))
        new_mask_corner2 = mask_obj_WSI.crop((corner2_coords[0],corner2_coords[1],corner2_coords[0]+patchSize[0],corner2_coords[1]+patchSize[1]))

        new_region_corner2.save(os.path.join(save_path,f'patches/patch_{k:04}_c2.png'))
        new_mask_corner2.save(os.path.join(save_path,f'masks/mask_{k:04}_c2.png'))

        corner3_coords = (coords[2] - patchSize[0], coords[3] - patchSize[1])
        new_region_corner3 = slide.read_region((int(corner3_coords[0]*factor), int(corner3_coords[1]*factor)), slide_dim_lvl, (patchSize[0], patchSize[1]))
        new_mask_corner3 = mask_obj_WSI.crop((corner3_coords[0],corner3_coords[1],corner3_coords[0]+patchSize[0],corner3_coords[1]+patchSize[1]))

        new_region_corner3.save(os.path.join(save_path,f'patches/patch_{k:04}_c3.png'))
        new_mask_corner3.save(os.path.join(save_path,f'masks/mask_{k:04}_c3.png'))

        corner4_coords = (coords[0], coords[3] - patchSize[1])
        new_region_corner4 = slide.read_region((int(corner4_coords[0]*factor), int(corner4_coords[1]*factor)), slide_dim_lvl, (patchSize[0], patchSize[1]))
        new_mask_corner4 = mask_obj_WSI.crop((corner4_coords[0],corner4_coords[1],corner4_coords[0]+patchSize[0],corner4_coords[1]+patchSize[1]))

        new_region_corner4.save(os.path.join(save_path,f'patches/patch_{k:04}_c4.png'))
        new_mask_corner4.save(os.path.join(save_path,f'masks/mask_{k:04}_c4.png'))

        k += 1


    # ## Color normalization: macenko-LS & vahadane-LS.

    # In[39]:


    # Check for patches full of glass --> review annotations.
    # plt.figure()
    if remove_patches:
        files = glob.glob(os.path.join(save_path,'patches/*.png'))
        for patch in files:
            img = np.array(Image.open(patch))[:,:,0:3]
            # print(f'min: {img[:,:,0:3].min()}, max: {img[:,:,0:3].max()}, mean: {img[:,:,0:3].mean()}, std: {img[:,:,0:3].std()}')
            if img[:,:,0:3].std() < 15:
                print(patch)
                os.remove(patch)
                os.remove(patch.replace('patches','masks').replace('patch','mask'))


    # In[40]:


    def normalize_alternative(image, ref, method, standartize_brightness):
        if standartize_brightness:
            image = staintools.LuminosityStandardizer.standardize(image)
            ref = staintools.LuminosityStandardizer.standardize(ref)
                
        if method == 'reinhard':
            normalizer = staintools.ReinhardColorNormalizer()
        if method == 'vahadane':
            normalizer = staintools.StainNormalizer(method='vahadane')
        if method == 'macenko':
            normalizer = staintools.StainNormalizer(method='macenko')
                
        normalizer.fit(ref)
        norm_img = normalizer.transform(image)
            
        return norm_img  


    # In[41]:


    '''
    This is taking a lot of time --> paralelize it! --> threads
    '''

    target = staintools.read_image("../data/norm_reference_512x512.png")
    folder_patches = '/Users/gabriel.jimenez/Documents/project/togitlab/dataset/512x512/' + slide_name.split('/')[-1].split('.ndpi')[0] + '/patches/'
    output_folder_macenko = '/Users/gabriel.jimenez/Documents/project/togitlab/dataset/512x512/' + slide_name.split('/')[-1].split('.ndpi')[0] + '/macenko/'
    output_folder_vahadane = '/Users/gabriel.jimenez/Documents/project/togitlab/dataset/512x512/' + slide_name.split('/')[-1].split('.ndpi')[0] + '/vahadane/'

    os.makedirs(output_folder_macenko, exist_ok=True)
    os.makedirs(output_folder_vahadane, exist_ok=True)

    for img in tqdm(glob.glob(folder_patches+'*.png')):
        # print(img)
        source = staintools.read_image(img)

        norm_img_macenko = normalize_alternative(source, target, 'macenko', True)
        output_filename_macenko = output_folder_macenko + img.split('/')[-1]
        norm_img_macenko = Image.fromarray(norm_img_macenko)
        norm_img_macenko.save(output_filename_macenko)
        
        #norm_img_vahadane = normalize_alternative(source, target, 'vahadane', True)
        #output_filename_vahadane = output_folder_vahadane + img.split('/')[-1]
        #norm_img_vahadane = Image.fromarray(norm_img_vahadane)
        #norm_img_vahadane.save(output_filename_vahadane)


    # ## Background extraction.
    # 
    # 

    # In[ ]:


    # size of region : cols, rows 
    def create_image_patches(img, num_rows, num_columns):
        """
        Partition an image into multiple patches of approximately equal size.
        The patch size is based on the desired number of rows and columns.
        Returns a list of image patches, in row-major order.
        """

        patch_list = []
        width, height = img.shape[1], img.shape[0]
        w, h = width // num_columns, height // num_rows # // is similar to the floor function
        #print(w,h)
        
        for y in range(0, h*num_rows, num_rows): 
            y_end = min(y + num_rows, height)
            for x in range(0, w*num_columns, num_columns):
                x_end = min(x + num_columns, width)
                patch = img[y:y_end, x:x_end]
                patch_list.append(patch)

        return patch_list


    # In[ ]:


    patchSize = 512

    background_path = '/Users/gabriel.jimenez/Documents/project/togitlab/dataset/512x512/' + slide_name.split('/')[-1].split('.ndpi')[0] + '/background_macenko/'
    os.makedirs(background_path, exist_ok=True)

    background_patch = []
    for coords_bbox, size_bbox in zip(coords_region, size):
        mask_ROI_WSI = mask_ROI.crop((coords_bbox[0],coords_bbox[1],coords_bbox[0]+size_bbox[0],coords_bbox[1]+size_bbox[1]))
        mask_obj_WSI = mask_obj.crop((coords_bbox[0],coords_bbox[1],coords_bbox[0]+size_bbox[0],coords_bbox[1]+size_bbox[1]))
        region = slide.read_region((int(coords_bbox[0]*factor), int(coords_bbox[1]*factor)), slide_dim_lvl, (size_bbox[0], size_bbox[1]))

        mask_ROI_WSI = (np.array(mask_ROI_WSI)>0).astype("uint8")
        mask_obj_WSI = (np.array(mask_obj_WSI)>0).astype("uint8")

        patch_ROI = create_image_patches(mask_ROI_WSI, patchSize, patchSize)
        patch_obj = create_image_patches(mask_obj_WSI, patchSize, patchSize)
        patch_region = create_image_patches(np.array(region), patchSize, patchSize)

        save_obj = []
        for i in tqdm(range(len(patch_ROI))):
            pixelSumROI = np.sum(patch_ROI[i])
            if pixelSumROI == patchSize*patchSize:
                pixelSumObj = np.sum(patch_obj[i])
                if pixelSumObj == 0:
                    if patch_region[i][:,:,0:3].std() > 20.0: # get only background with few pixels of glass.
                        background_patch.append(patch_region[i])

                        bg_patch_norm = normalize_alternative(patch_region[i], target, 'macenko', True)
                        Image.fromarray(bg_patch_norm).save(os.path.join(background_path,f'bg_{i:04}.png'))

    print(len(background_patch))
            


    # In[ ]:


    # plt.figure()
    # i = 0
    # for img in background_patch:
    #     print(img[:,:,0:3].shape, i)
    #     print(f'min: {img[:,:,0:3].min()}, max: {img[:,:,0:3].max()}, mean: {img[:,:,0:3].mean()}, std: {img[:,:,0:3].std()}')
    # #     # plt.imshow(img[:,:,0:3])
    # #     # plt.show()
    #     i += 1


    # In[ ]:


    # print(coords_region)
    # plt.figure()
    # plt.imshow(mask_ROI_WSI)
    # plt.show()


    # - Create CSV --> datasetID, WSI_name, WSI_id, CN_id (color normalization), patchID, coords of patch, BOR (background to object ratio)
    # - patchID --> WSIid_patch_nnnn_cX.npy(png) --> c0: object center, c1: top-left, c2: top-right, c3: bottom-right, c4: bottom-left
    # - CN_id --> 1: something, CN_id --> 2: else
