# Code to extract OpenLiDAR 3D Object Detection Dataset into KITTI Format

import os
import numpy as np
import argparse
import open3d as o3d
import numpy as np
from typing import Union
import pickle

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--path_to_openlidar_dataset', type=str, default=None, help='Path to OpenLiDAR Dataset')

    parser.add_argument('--path_to_openlidar_kitti_format', type=int, default=None, required=True, help='Path to Result of OpenLiDAR Dataset in KITTI Format')
    #parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')

def read_pcd(path: str) -> np.ndarray:
    """
    Reads a pointcloud with open3d and returns it as a numpy ndarray
    Args:
        path (str): path to the pointcloud to be read or the directory containing it/them
    Returns:
        np.ndarray: the pointcloud(s) as a numpy ndarray (dims: (pointcloud x) points x coordinates)
    """
    if os.path.isdir(path):
        pointclouds = []
        filenames = os.listdir(path)
        for filename in filenames:
            if filename[-4:] != '.pcd':
                
                if filename[-4 : ] == ".npy" :
                    
                    pcd = np.load( path )
                    pointclouds.append( np.array( pcd ))
                    
                else :
                    continue
            else :
                
                pcd = o3d.io.read_point_cloud(path)
                pointclouds.append(np.asarray(pcd.points))
        return np.array(pointclouds)
    
    elif os.path.isfile(path):
        
        filename = path.split("/")[-1]
        
        if filename[ -4 : ] == ".pcd" :
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
        elif filename[ -4 : ] == ".npy" :
            pcd = np.load( path )
            return pcd
        


def pcd_to_bin(pcd: Union[str, o3d.geometry.PointCloud, np.ndarray], out_path: str , prefix_file_name : str = None ):
    """
    Convert pointcloud from '.pcd' to '.bin' format
    Args:
        pcd (Union[str, PointCloud, np.ndarray]): the pointcloud to be converted (either its path, or the pointcloud itself)
        out_path (str): the path to the destination '.bin' file
    """
    # if 'pcd' is a string, assume it's the path to the pointcloud to be converted
    if isinstance(pcd, str):
        pcd = read_pcd(path=pcd)
    # save the poinctloud to '.bin' format
    out_path += "" if out_path[-4:] == ".bin" else ".bin"
    if prefix_file_name is not None :
        
        out_path = "/".join( out_path.split("/")[ : -1] ) + "/" + prefix_file_name + out_path.split("/")[-1]
        
    np.asarray(pcd).astype('float32').tofile(out_path)


def convert_all(in_dir_path: str, out_dir_path: str , prefix_file_name = None ):
    """
    Converts all the pointclouds in the specified drectory from '.pcd' to '.bin' format
    Args:
        in_dir_path (str): path of the directory containing the pointclouds tio be converted
        out_dir_path (str): path of the directory where to put the resulting converted pointclouds
    Raises:
        ValueError:
            if 'in_dir_path' is neither a regular file nor a directory;
            if 'out_dir_path' is not a directory.
    Returns:
        np.ndarray: the converted pointclouds as numpy ndarrays
    """
    
    os.makedirs( out_dir_path , exist_ok= True )
    
    if not os.path.isdir(out_dir_path):
        raise ValueError(f"{out_dir_path} is not a directory.")

    if os.path.isfile(in_dir_path):
        
        if prefix_file_name is None :
            file_name = in_dir_path.split('/')[-1]
        else :
            file_name = prefix_file_name + in_dir_path.split('/')[-1]
            
        for filename in filenames:
            if filename[-4:] != '.pcd':
                continue
            out_path = os.path.join(out_dir_path, filename[:-4] + '.bin')
            pcd_to_bin(pcd=os.path.join(in_dir_path, filename), out_path=out_path)
            
    elif os.path.isdir(in_dir_path):
        pointclouds = []
        filenames = sorted( os.listdir(in_dir_path))
        for filename in filenames:
            #print( "Converting file LiDAR point cloud : " + str( filename ))
            if ( filename[-4:] != '.pcd' ) & ( filename[ -4 : ] != ".npy" ):
                continue
                
            #if prefix_file_name is None :
            #    filename = filename
            #elif prefix_file_name is not None :
            #    filename = prefix_file_name + filename
            out_path = os.path.join(out_dir_path, filename[:-4] + '.bin')
            pcd_to_bin(pcd=os.path.join(in_dir_path, filename), out_path=out_path , prefix_file_name = prefix_file_name )

    else:
        raise ValueError(f"'{in_dir_path} is neither a directory or file")

def main():
    args, cfg = parse_config()

    FILE_OF_TRAIN_OPENLIDAR_DATASET_PATH = args.path_to_openlidar_dataset #"/media/ofel04/A2/240718_LiDAR benchmark/train_2022-07-30-09_01_00_01_00_00_02/"

    FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH = args.path_to_openlidar_kitti_format # FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH = "/media/ofel04/A2/240718_LiDAR benchmark/OpenLiDAR_KITTI_Format/"

    # Create folder for Result OpenLiDAR Dataset KITTI Format

    if os.path.exists( FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH ) == False :

        os.makedirs( FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH )

        print( "Making Folder OpenLiDAR Dataset with KITTI Format in path : " + str( FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH ))

    # Make folder for OpenLiDAR KITTI Dataset

    FOLDER_FOR_OPENLIDAR_TRAIN_VALIDATION_SPLIT_PATH = FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH + "ImageSets/"

    FOLDER_FOR_OPENLIDAR_TRAINING_IN_KITTI_FORMAT_PATH = FOLDER_OF_OPENLIDAR_DATASET_KITTI_FORMAT_PATH + "training/"

    FOLDER_FOR_OPENLIDAR_VELODYNE_TRAINING_IN_KITTI_FORMAT = FOLDER_FOR_OPENLIDAR_TRAINING_IN_KITTI_FORMAT_PATH + "velodyne/"

    FOLDER_FOR_OPENLIDAR_LABEL_TRAINING_IN_KITTI_FORMAT = FOLDER_FOR_OPENLIDAR_TRAINING_IN_KITTI_FORMAT_PATH + "label/"

    os.makedirs( FOLDER_FOR_OPENLIDAR_TRAIN_VALIDATION_SPLIT_PATH , exist_ok= True )

    os.makedirs( FOLDER_FOR_OPENLIDAR_TRAINING_IN_KITTI_FORMAT_PATH , exist_ok = True )

    os.makedirs( FOLDER_FOR_OPENLIDAR_VELODYNE_TRAINING_IN_KITTI_FORMAT , exist_ok = True )

    os.makedirs( FOLDER_FOR_OPENLIDAR_LABEL_TRAINING_IN_KITTI_FORMAT , exist_ok= True )


    # Convert OpenLiDAR Point Cloud into KITTI Format

    convert_all( in_dir_path = FILE_OF_TRAIN_OPENLIDAR_DATASET_PATH,
            out_dir_path= FOLDER_FOR_OPENLIDAR_VELODYNE_TRAINING_IN_KITTI_FORMAT,
            prefix_file_name = "00")

    # Extract ground truth label of OpenLiDAR Point Cloud into KITTI Format

    dataset_path = None

    for file_openlidar_dataset in os.listdir( FILE_OF_TRAIN_OPENLIDAR_DATASET_PATH ) :

        if file_openlidar_dataset[ -4 : ] == ".pkl" :

            dataset_path = FILE_OF_TRAIN_OPENLIDAR_DATASET_PATH + file_openlidar_dataset 

            print( "Extract ground truth 3D Object Detection OpenLiDAR dataset using dictionary file : " + str( dataset_path ))

    assert dataset_path is not None, "Didnt found ground truth 3D object Detection for OpenLiDAR Dataset in folder : " + str( FILE_OF_TRAIN_OPENLIDAR_DATASET_PATH )

    with open( dataset_path , "rb+" ) as f :

        dataset_open_lidar = pickle.load( f )
        
    for lidar_scene_idx, lidar_scene_label in enumerate( dataset_open_lidar ) :
        
        #print( "LiDAR scene label is : " + str( lidar_scene_label ))
        
        name_of_label_file = "00" + str( lidar_scene_label[ "frame_id" ] ).split("_")[-1] + ".txt"
        
        for object_in_lidar_scene_idx in range( len( lidar_scene_label[ "annos" ][ "name" ])) :
            
            label_for_object_in_lidar_scene = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format( lidar_scene_label["annos"]["name" ][ object_in_lidar_scene_idx ] if lidar_scene_label["annos"]["name" ][ object_in_lidar_scene_idx ] != "other vehicles" else "other_vehicles",
                                                                                                    0,
                                                                                                    lidar_scene_label["annos"]["difficulty" ][ object_in_lidar_scene_idx ],
                                                                                                    0,
                                                                                                    -100,
                                                                                                    -100,
                                                                                                    -100,
                                                                                                    -100,
                                                                                                    lidar_scene_label["annos"]["dimensions" ][ object_in_lidar_scene_idx ][0],
                                                                                                    lidar_scene_label["annos"]["dimensions" ][ object_in_lidar_scene_idx ][1],
                                                                                                    lidar_scene_label["annos"]["dimensions" ][ object_in_lidar_scene_idx ][2],
                                                                                                    lidar_scene_label["annos"]["location" ][ object_in_lidar_scene_idx ][0],
                                                                                                    lidar_scene_label["annos"]["location" ][ object_in_lidar_scene_idx ][1],
                                                                                                    lidar_scene_label["annos"]["location" ][ object_in_lidar_scene_idx ][2],
                                                                                                    lidar_scene_label["annos"]["heading_angles" ][ object_in_lidar_scene_idx ] )
            
            with open( FOLDER_FOR_OPENLIDAR_LABEL_TRAINING_IN_KITTI_FORMAT + name_of_label_file , "a+" ) as f :
                
                f.write( label_for_object_in_lidar_scene )
                

        print( "Finished writing labels for scene {} to LiDAR scenes annotation file {}".format( str( lidar_scene_idx + 1 ) , FOLDER_FOR_OPENLIDAR_LABEL_TRAINING_IN_KITTI_FORMAT + name_of_label_file))


    # Extract Training and Validation Split of OpenLiDAR Dataset in KITTI Format

    number_of_validation_scenes_per_files_in_dataset = 2 #2

    for open_lidar_lidar_index, open_lidar_lidar_scene in enumerate( sorted( os.listdir( FOLDER_FOR_OPENLIDAR_VELODYNE_TRAINING_IN_KITTI_FORMAT ) )) :
        
        name_of_lidar_scene = open_lidar_lidar_scene.replace( ".bin" , "")
        
        #print( "Scene of Open LiDAR scene is : " + str( name_of_lidar_scene ))

        
        with open( FOLDER_FOR_OPENLIDAR_TRAIN_VALIDATION_SPLIT_PATH + "train.txt" , "a+" ) as f :
            
            f.write( name_of_lidar_scene + "\n" )
        
        with open( FOLDER_FOR_OPENLIDAR_TRAIN_VALIDATION_SPLIT_PATH + "trainval.txt" , "a+" ) as f :
            
            f.write( name_of_lidar_scene + "\n" )
            
        if open_lidar_lidar_index % number_of_validation_scenes_per_files_in_dataset == 0 :
            
            with open( FOLDER_FOR_OPENLIDAR_TRAIN_VALIDATION_SPLIT_PATH + "val.txt", "a+" ) as f :
                
                f.write( name_of_lidar_scene + "\n" )
        
        """        
        with open( FOLDER_FOR_OPEN_LIDAR_TRAIN_TEST_SPLIT + "test.txt", "a+" ) as f :

            f.write( name_of_lidar_scene + "\n" )
        """


if __name__ == '__main__':
    main()
    



    


