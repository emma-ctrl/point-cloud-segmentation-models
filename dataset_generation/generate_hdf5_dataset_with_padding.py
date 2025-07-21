# This code is used to import raw colored point cloud in .pcd format for a scanned object
# and generate a fixed size dataset by supplying the "desired_number_of_sample_points"
# which in the example below fixed to 4096 colored points per sample. This is done by
# random downsampling the point cloud if the number of points available is greater than
# the desired or otherwise by zero padding.
# Each object has 1200 samples
# The EAN code (bar code) of each scanned object is provided 

import open3d as o3d
import os
import glob
import numpy as np
import h5py
from matplotlib import pyplot as plt
import random
import getpass

print(o3d.__version__)


# https://www.ean-search.org

def generate_hdf5_dataset_with_padding(path, hdf5_filename, desired_number_of_sample_points):
    # Get files
    pcds = sorted(glob.glob(os.path.join(path, "*.pcd")))
    print(f"🔍 Found {len(pcds)} .pcd files in {path}")

    if len(pcds) == 0:
        print(f" ERROR: No PCD files found in {path}. Skipping.")
        return

    # Open HDF5 file in write mode
    with h5py.File(hdf5_filename, 'w') as f:
        point_clouds = []
        color_clouds = []

        for i, pcd in enumerate(pcds):
            # Load the point cloud
            cloud = o3d.io.read_point_cloud(pcd)
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors)

            print(f"📌 Processing {pcd}: Found {points.shape[0]} points.")

            if points.shape[0] == 0:
                print(f"WARNING: Empty point cloud in {pcd}, skipping this file.")
                continue

            if points.shape[0] > desired_number_of_sample_points:
                random_index = random.sample(range(points.shape[0]), desired_number_of_sample_points)
                downsampled_points = points[random_index]
                downsampled_colors = colors[random_index]
            else:
                # Pad with zeros if needed
                pad_amount = desired_number_of_sample_points - points.shape[0]
                downsampled_points = np.pad(points, ((0, pad_amount), (0, 0)), 'constant', constant_values=(0, 0))
                downsampled_colors = np.pad(colors, ((0, pad_amount), (0, 0)), 'constant', constant_values=(0, 0))

            point_clouds.append(downsampled_points)
            color_clouds.append(downsampled_colors)

        # Check if any valid data was collected
        if len(point_clouds) == 0:
            print(f"ERROR: No valid point cloud data found in {path}. Skipping save.")
            return

        # Convert lists to numpy arrays
        point_clouds = np.asarray(point_clouds)
        color_clouds = np.asarray(color_clouds)

        # Save to HDF5
        f.create_dataset("point_clouds", data=point_clouds)
        f.create_dataset("color_clouds", data=color_clouds)

    print(f"Successfully saved: {hdf5_filename} with shape {point_clouds.shape}")

NUM_SAMPLES = 1200
NUM_POINTS = 2048

source_dir = "D:/Emma/MiniMarket_Emma/datasets/MiniMarket_raw"
save_to_dir = "D:/Emma/MiniMarket_Emma/datasets/MiniMarket_processed_dataset"


#creating output directory if it doesn't exist
if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

#1 COFFEE
EAN_code= "8711000519325"
EAN_product_name= "Smooth"
rec_grasp_force= "40" # in Newton
weight= "365" # in grams
object_name= "coffee_kenco_100gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "7613038575650"
EAN_product_name= "Nescafe original 3 in 1"
rec_grasp_force= "07" # in Newton
weight= "127" # in grams
object_name= "coffee_nescafe_3in1_original_6cups"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5039303005004"
EAN_product_name= "Costa Coffee Smooth Medium Roast"
rec_grasp_force= "30" # in Newton
weight= "155" # in grams
object_name= "coffee_instant_costa_coffee_smooth_medium_roast_100gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8711000519141"
EAN_product_name= "Kenco Millicano Americano Instant Coffee 100g"
rec_grasp_force= "30" # in Newton
weight= "155" # in grams
object_name= "coffee_instant_kenco_millicano_americano_100gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8445290044785"
EAN_product_name= "Nescafe Gold Blend Roastery Collection"
rec_grasp_force= "40" # in Newton
weight= "154" # in grams
object_name= "coffee_nescafe_gold_blend_roastery_collection_95gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )





#2 Liquid Handwash
EAN_code= "8714789673653"
EAN_product_name= "Palmolive Hygiene Plus Sensitive Liquid Handwash, 300ml"
rec_grasp_force= "40" # in Newton
weight= "344" # in grams
object_name= "liquid_handwash_palmolive_hygiene_plus_sensitive_300ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8003520013026"
EAN_product_name= "Palmolive Milk And Honey Hand Soap 300 ml"
rec_grasp_force= "40" # in Newton
weight= "344" # in grams
object_name= "liquid_handwash_palmolive_milk_and_honey_300ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8003520023322"
EAN_product_name= "Palmolive Hand Wash Antibacterial"
rec_grasp_force= "40" # in Newton
weight= "344" # in grams
object_name= "liquid_handwash_palmolive_antibacterial_300ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101346170"
EAN_product_name= "Carex Original Hand Wash 250 ml"
rec_grasp_force= "40" # in Newton
weight= "292" # in grams
object_name= "liquid_handwash_carex_original_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101959509"
EAN_product_name= "Carex Moisture Plus Hand Wash 250 ml"
rec_grasp_force= "40" # in Newton
weight= "292" # in grams
object_name= "liquid_handwash_carex_moisture_plus_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "4000388177000"
EAN_product_name= "Dove Caring Hand Wash 250ml"
rec_grasp_force= "40" # in Newton
weight= "288" # in grams
object_name= "liquid_handwash_dove_moisturising_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8720181049361"
EAN_product_name= "Dove Care & Protect Hand Wash 250ml"
rec_grasp_force= "40" # in Newton
weight= "288" # in grams
object_name= "liquid_handwash_dove_care_and_protect_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







#3 HOT CHOCOLATE
EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/hot_choc_cadbury_250gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/hot_choc_cadbury_250gm_1200_2048",NUM_POINTS)

EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/hot_choc_instant_cadbury_400gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/hot_choc_instant_cadbury_400gm_1200_2048",NUM_POINTS)

EAN_code= "5034660021445"
EAN_product_name= "Cadbury Fair Trade Bournville Cocoa 250g"
rec_grasp_force= "30" # in Newton
weight= "295" # in grams
object_name= "hot_choc_cadbury_bournville_cocoa_ideal_for_baking_250gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/hot_choc_instant_galaxy_370gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/hot_choc_instant_galaxy_370gm_1200_2048",NUM_POINTS)



#4 HONEY
EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/honey_rowse_340gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/honey_rowse_340gm_1200_2048",NUM_POINTS)





#5 JAM
EAN_code= "5060391624495"
EAN_product_name= "Best of apricot"
rec_grasp_force= "40" # in Newton
weight= "493" # in grams
object_name= "jam_hartleys_apricot_300gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5060391624518"
EAN_product_name= "Hartleys seedless strawberry jam"
rec_grasp_force= "40" # in Newton
weight= "493" # in grams
object_name= "jam_hartleys_strawberry_300gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5060391625119"
EAN_product_name= "Hartley's Pineapple Jam"
rec_grasp_force= "40" # in Newton
weight= "493" # in grams
object_name= "jam_hartleys_pineapple_300gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "3045320094084"
EAN_product_name= "Bonne Maman raspberry conserve"
rec_grasp_force= "45" # in Newton
weight= "585" # in grams
object_name= "jam_bonne_maman_raspberry_conserve_370gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "3045320094053"
EAN_product_name= "Bonne Maman orange fine marmalade"
rec_grasp_force= "45" # in Newton
weight= "585" # in grams
object_name= "jam_bonne_maman_bitter_orange_fine_marmalade_370gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )






#6 CEREALS
EAN_code= "5053827209762"
EAN_product_name= "Kelloggs Crunchy Nut"
rec_grasp_force= "10" # in Newton
weight= "355" # in grams
object_name= "cereal_kelloggs_crunchy_nut_honey_and_nut_300gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "7613287087713"
EAN_product_name= "Nesquick chocolate pillows"
rec_grasp_force= "10" # in Newton
weight= "400" # in grams
object_name= "cereal_nesquick_chocolate_and_banana_pillows_350gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000108031192"
EAN_product_name= "2 x Quaker Oat So Simple Simply Strawberry Porridge No Added Sugar Sachets 8 per pack"
rec_grasp_force= "10" # in Newton
weight= "315" # in grams
object_name= "cereal_quaker_oat_so_simple_simply_strawberry_porridge_8_sachets_260gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000108030553"
EAN_product_name= "Quaker Oats Quaker Oat So Simple golden syrup porridge cereal sachets"
rec_grasp_force= "10" # in Newton
weight= "432" # in grams
object_name= "cereal_quaker_oat_so_simple_golden_syrup_porridge_10_sachets_360gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )






#7 MICROWAVE RICE
EAN_code= "5010034012110"
EAN_product_name= "CAJUN SPICED RICR"
rec_grasp_force= "10" # in Newton
weight= "261" # in grams
object_name= "microwave_rice_bens_original_cajun_spiced_250gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5010034013681"
EAN_product_name= "teolemon"
rec_grasp_force= "10" # in Newton
weight= "230" # in grams
object_name= "microwave_rice_bens_original_basmati_classic_220gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5010034013674"
EAN_product_name= "null"
rec_grasp_force= "10" # in Newton
weight= "226" # in grams
object_name= "microwave_rice_bens_original_long_grain_classic_220gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5011157888101"
EAN_product_name= "Tilda steamed pure basmati rice"
rec_grasp_force= "10" # in Newton
weight= "259" # in grams
object_name= "microwave_rice_tilda_steamed_pure_basmati_rice_250gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5011157888132"
EAN_product_name= "Tilda steamed brown basmati rice"
rec_grasp_force= "10" # in Newton
weight= "254" # in grams
object_name= "microwave_rice_tilda_steamed_wholegrain_basmati_rice_250gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







#8 SOAP BAR
EAN_code= "8720181218224"
EAN_product_name= "Dove Original Beauty Cream Bar 90 g"
rec_grasp_force= "07" # in Newton
weight= "100" # in grams
object_name= "soap_bar_dove_original_beauty_cream_bar_1x_90gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8720181218279"
EAN_product_name= "Dove Original Beauty Bar 90 g x 2"
rec_grasp_force= "10" # in Newton
weight= "200" # in grams
object_name= "soap_bar_dove_original_beauty_cream_bar_2x_90gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8720181218286"
EAN_product_name= "Dove Original Beauty Bar 90 g x 4"
rec_grasp_force= "20" # in Newton
weight= "400" # in grams
object_name= "soap_bar_dove_original_beauty_cream_bar_4x_90gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8720181218293"
EAN_product_name= "Dove Original Beauty Bar 90 g x 6"
rec_grasp_force= "30" # in Newton
weight= "600" # in grams
object_name= "soap_bar_dove_original_beauty_cream_bar_6x_90gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8901030024061"
EAN_product_name= "Pears Transparent Oil Clear Soap 125g"
rec_grasp_force= "07" # in Newton
weight= "145" # in grams
object_name= "soap_bar_pears_transparent_soap_lemon_flower_extracts_125gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101082566"
EAN_product_name= "Imperial Leather Original Bar Soap 4 x 100g"
rec_grasp_force= "20" # in Newton
weight= "408" # in grams
object_name= "soap_bar_imperial_leather_original_4x_100gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101082214"
EAN_product_name= "Imperial Leather Gentle Bar Soap 4 x 100g"
rec_grasp_force= "20" # in Newton
weight= "408" # in grams
object_name= "soap_bar_imperial_leather_gentle_care_4x_100gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5025404001274"
EAN_product_name= "Sheild Shield Soap Aqua 4pk"
rec_grasp_force= "25" # in Newton
weight= "468" # in grams
object_name= "soap_bar_sheild_fresh_aqua_deodorising_4x_115gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )





# CONDIMENTS
EAN_code= "5019989103096"
EAN_product_name= "Saxa Fine Sea Salt 350g"
rec_grasp_force= "30" # in Newton
weight= "396" # in grams
object_name= "condiments_salt_saxa_fine_sea_salt_350gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5010024111069"
EAN_product_name= "Saxa table salt"
rec_grasp_force= "40" # in Newton
weight= "804" # in grams
object_name= "condiments_salt_saxa_fine_salt_750gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# TINNED SOUP
EAN_code= "5000157146816"
EAN_product_name= "none"
rec_grasp_force= "40" # in Newton
weight= "453" # in grams
object_name= "tinned_soup_heinz_creamy_tomato_soup_plant_based_400gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# TOMATO SAUCE
EAN_code= "4002359015373"
EAN_product_name= "Dolmio 7 Vegetables Tomato & Chilli Pasta Sauce 350g"
rec_grasp_force= "40" # in Newton
weight= "564" # in grams
object_name= "tomato_sauce_dolmio_7_vegetables_tomato_and_chilli_pasta_sauce_350gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "4002359004841"
EAN_product_name= "Dolmio Bolognese Smooth Tomato Pasta Sauce 750ml"
rec_grasp_force= "50" # in Newton
weight= "1074" # in grams
object_name= "tomato_sauce_dolmio_bolognese_smooth_tomato_pasta_sauce_750gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "4002359014673"
EAN_product_name= "chilli con carne medium"
rec_grasp_force= "40" # in Newton
weight= "692" # in grams
object_name= "tomato_sauce_bens_original_chilli_con_carne_medium_450gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000157146298"
EAN_product_name= "Sun-dried cherry tomato and basil"
rec_grasp_force= "40" # in Newton
weight= "528" # in grams
object_name= "tomato_sauce_heinz_sun_dried_cherry_tomato_and_basil_350gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# SHOWER GEL
EAN_code= "5000101500183"
EAN_product_name= "Original Source Rhubarb and Raspberry Shower 250ml"
rec_grasp_force= "30" # in Newton
weight= "290" # in grams
object_name= "shower_gel_original_source_rhubarb_and_raspberry_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101100468"
EAN_product_name= "Original Source Coconut (White) Shower 250ml"
rec_grasp_force= "30" # in Newton
weight= "286" # in grams
object_name= "shower_gel_original_source_coconut_and_shea_butter_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000101845208"
EAN_product_name= "Orig Source Shower Gel Mint"
rec_grasp_force= "30" # in Newton
weight= "292" # in grams
object_name= "shower_gel_original_source_mint_and_tea_tree_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )









# SHOWER CREAM
EAN_code= "8714789732923"
EAN_product_name= "Palmolive Naturals Coconut Shower Gel"
rec_grasp_force= "30" # in Newton
weight= "290" # in grams
object_name= "shower_cream_palmolive_naturals_coconut_and_milk_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8714789732879"
EAN_product_name= "Palmolive Naturals Shower Cream"
rec_grasp_force= "30" # in Newton
weight= "290" # in grams
object_name= "shower_cream_palmolive_naturals_milk_and_honey_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# MOISTURISING CREAM
EAN_code= "4005808890576"
EAN_product_name= "Nivea Soft Moisturising Cream"
rec_grasp_force= "15" # in Newton
weight= "088" # in grams
object_name= "moisturising_cream_nivea_soft_with_jojoba_oil_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# HAIR AND BODY WASH
EAN_code= "5029066007155"
EAN_product_name= "Child's Farm Childs Farm Hair & Body Wash 250ml"
rec_grasp_force= "30" # in Newton
weight= "294" # in grams
object_name= "hair_and_body_wash_childs_farm_organic_sweet_orange_250ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )






# CONDITIONER
EAN_code= "3600524074807"
EAN_product_name= "Loreal LOral Paris Elvive Bond Repair Conditioner 150ml"
rec_grasp_force= "15" # in Newton
weight= "165" # in grams
object_name= "conditioner_loreal_paris_elvive_bond_repair_conditioner_150ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# PASTA
EAN_code= "5000232024442"
EAN_product_name= "Penne"
rec_grasp_force= "20" # in Newton
weight= "535" # in grams
object_name= "pasta_napolina_penne_no_50_500gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000232027306"
EAN_product_name= "Lasagna sheets"
rec_grasp_force= "20" # in Newton
weight= "401" # in grams
object_name= "pasta_napolina_lasagna_sheets_375gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000232024428"
EAN_product_name= "Fusilli"
rec_grasp_force= "20" # in Newton
weight= "534" # in grams
object_name= "pasta_napolina_fusilli_no_56_500gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5000232024725"
EAN_product_name= "Rigatoni"
rec_grasp_force= "20" # in Newton
weight= "527" # in grams
object_name= "pasta_napolina_rigatoni_no_38_500gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# BISCUITS
EAN_code= "7622210740458"
EAN_product_name= "Golden Oats"
rec_grasp_force= "10" # in Newton
weight= "262" # in grams
object_name= "biscuits_belvita_breakfast_biscuits_golden_oats_5_pack_225gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )






# TEA
EAN_code= "5000208046607"
EAN_product_name= "none"
rec_grasp_force= "05" # in Newton
weight= "62" # in grams
object_name= "tea_tetley_discovery_black_tea_with_lemon_and_ginger_20_bags_46gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "070177264574"
EAN_product_name= "Twinings Lemon Green Tea - 20 Tea Bags"
rec_grasp_force= "05" # in Newton
weight= "60" # in grams
object_name= "tea_twinings_lemon_green_tea_20_bags_40gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5010357112009"
EAN_product_name= "Yorkshire Teabags 40s 125g"
rec_grasp_force= "07" # in Newton
weight= "152" # in grams
object_name= "tea_taylors_of_harrogate_yorkshire_tea_40_bags_125gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )







# NAIL POLISH REMOVER
EAN_code= "309971370281"
EAN_product_name= "Cutex Ultra- Powerful Nail Polish Remover 200ml"
rec_grasp_force= "30" # in Newton
weight= "203" # in grams
object_name= "nail_polish_remover_cutex_ultra_powerful_200ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "309971370212"
EAN_product_name= "Cutex Nourishing Nail Polish Remover 100ml"
rec_grasp_force= "30" # in Newton
weight= "113" # in grams
object_name= "nail_polish_remover_cutex_nourishing_100ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "309971370274"
EAN_product_name= "Cutex Ultra- Powerful Nail Polish Remover 100ml"
rec_grasp_force= "30" # in Newton
weight= "110" # in grams
object_name= "nail_polish_remover_cutex_ultra_powerful_100ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )









#9 DISH WASH LIQUID
EAN_code= "8006540723432"
EAN_product_name= "Fairy Original Washing Up Liquid 383 ml"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/dishwash_fairy_original_383ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/dishwash_fairy_original_383ml_1200_2048",NUM_POINTS)

EAN_code= "8006540994474"
EAN_product_name= "null"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/dishwash_fairy_lemon_320ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/dishwash_fairy_lemon_320ml_1200_2048",NUM_POINTS)



#10 KETCHUP
EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/ketchup_heinz_400ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/ketchup_heinz_400ml_1200_2048",NUM_POINTS)



#11 SPREADS
EAN_code= "5410126116953"
EAN_product_name= "Lotus Biscuit Spread Smooth 400g"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/biscuit_spread_lotus_400gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/biscuit_spread_lotus_400gm_1200_2048",NUM_POINTS)

EAN_code= ""
EAN_product_name= ""
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/hazelnut_cocoa_spread_nutella_350gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/hazelnut_cocoa_spread_nutella_350gm_1200_2048",NUM_POINTS)

EAN_code= "80135463"
EAN_product_name= "none"
rec_grasp_force= "40" # in Newton
weight= "381" # in grams
object_name= "hazelnut_cocoa_spread_nutella_200gm"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )


EAN_code= "5060391623092"
EAN_product_name= "Sun-Pat Crunchy"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/crunchy_peanut_butter_sunpat_400gm_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/crunchy_peanut_butter_sunpat_400gm_1200_2048",NUM_POINTS)






#12 SHAMPOO
EAN_code= "8006540810538"
EAN_product_name= "Head & Shoulders Classic Clean Clarifying Anti Dandruff Shampoo For Itchy, Dry Scalp And Hair 400ml"
rec_grasp_force= "40" # in Newton
weight= "457" # in grams
object_name= "shampoo_head_and_shoulders_classic_400ml_1200_2048"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "4084500015012"
EAN_product_name= "Head & Shoulders Classic Clean 2in1 Shampoo & Conditioner"
rec_grasp_force= "30" # in Newton
weight= "264" # in grams
object_name= "shampoo_head_and_shoulders_classic_225ml_1200_2048"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "5410076229468"
EAN_product_name= "Head & Shoulders Citrus Fresh Anti- Dandruff Shampoo"
rec_grasp_force= "30" # in Newton
weight= "290" # in grams
object_name= "shampoo_head_and_shoulders_citrus_250ml_1200_2048"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "8006540810729"
EAN_product_name= "Head & Shoulders Citrus Fresh 2in1 Anti Dandruff Shampoo And Conditioner For Greasy Hair 400ml"
rec_grasp_force= "40" # in Newton
weight= "457" # in grams
object_name= "shampoo_head_and_shoulders_citrus_400ml_1200_2048"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )

EAN_code= "3600523586417"
EAN_product_name= "L'Oréal Paris Elvive Dream Lengths Long Hair Shampoo 500ml"
rec_grasp_force= "40" # in Newton
weight= "564" # in grams
object_name= "shampoo_loreal_paris_elvive_dream_lengths_long__damaged_hair_500ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )






#13 TOOTH PASTE
EAN_code= "8718951099135"
EAN_product_name= "Max White Optic Toothpaste"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/tooth_paste_colgate_max_white_optic_75ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/tooth_paste_colgate_max_white_optic_75ml_1200_2048",NUM_POINTS)

EAN_code= "8718951166127"
EAN_product_name= "Colgate Max Fresh Cooling Crystals Toothpaste, 75 ml"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/tooth_paste_colgate_max_fresh_cooling_crystals_75ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/tooth_paste_colgate_max_fresh_cooling_crystals_75ml_1200_2048",NUM_POINTS)

EAN_code= "8714789710624"
EAN_product_name= "Colgate-Palmolive Colgate Tooth Paste Total Advanced Freshening"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/tooth_paste_colgate_total_active_fresh_125ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/tooth_paste_colgate_total_active_fresh_125ml_1200_2048",NUM_POINTS)

EAN_code= "5054563087195"
EAN_product_name= "Aquafresh Kids Toothpaste Splash Strawberry & Mint Flavour 3-8 Years 75ml"
rec_grasp_force= "07" # in Newton
weight= "" # in grams
object_name= "tooth_paste_aquafresh_kids_splash_strawberry_3_to_8_years_75ml"
#generate_hdf5_dataset_with_padding( source_dir + object_name +"/", save_to_dir + object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code, NUM_POINTS )


EAN_code= "8001090291035"
EAN_product_name= "Oral- B 3D White Arctic Fresh Toothpaste 75ml"
#generate_hdf5_dataset_with_padding("/home/"+getpass.getuser()+"/grasping_edi/tooth_paste_oral_B_3D_white_arctic_fresh_75ml_1200/","/home/"+getpass.getuser()+"/machine_learning/object_dataset/tooth_paste_oral_B_3D_white_arctic_fresh_75ml_1200_2048",NUM_POINTS)

#alternative to calling the function for each object individually 
#Get all subdirectories in the raw_pcd folder
object_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

#Process each object
print(f"Found {len(object_dirs)} objects to process.")
for object_name in object_dirs:
    print(f"Processing {object_name}...")
    #Use empty strings for metadata we don't have
    EAN_code = ""
    
    #Generate the output filename
	
    output_file = os.path.join(save_to_dir, object_name + "_" + str(NUM_SAMPLES) + "_" + str(NUM_POINTS) + "_" + EAN_code)
    
    #Process the object
    generate_hdf5_dataset_with_padding(os.path.join(source_dir, object_name), output_file, NUM_POINTS)
    
    print(f"Completed processing {object_name}")

print("All objects processed successfully")


#cloud = o3d.io.read_point_cloud("/home/"+getpass.getuser()+"/grasping_edi/tin_can_up/angle_32_light_4_noise_1_cam7.pcd")
#points= np.asarray(cloud)
#colors = np.asarray(cloud.colors)
#print(cloud)
#print(points)
#print(colors)
#o3d.visualization.draw_geometries(cloud)

#fig = plt.figure(figsize=(5, 5))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(points[:, 0], points[:, 1], points[:, 2])
#ax.set_axis_off()
#plt.show()
