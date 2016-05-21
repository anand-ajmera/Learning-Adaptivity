#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <fstream>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>

using namespace std;

int size_descriptor = 125;
string data_folder = "/home/anand/robocup_ws/src/mas_datasets/generic/mds_pointclouds/objects/workspace_setups/at_home/";
string object_names[4] = {"MuscleBox","BigCoffeeCup","Pringles","SmallKetchupBottle"};
string str;

pcl::PointCloud<pcl::PointXYZ>::Ptr load_pcd()
{  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    str.append(data_folder);
    str.append(object_names[1]);
    str.append("/5.pcd");

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (str, *cloud) == -1) //* load the file
    {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    // return (-1);
    }
    std::cout << "Loaded ";
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr perform_downsampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
 
    // Filter object.
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud);
    // We set the size of every voxel to be 1x1x1cm
    // (only one point per every cubic centimeter will survive).
    filter.setLeafSize(0.01f, 0.01f, 0.01f);
 
    filter.filter(*filteredCloud);

    return filteredCloud;
}

// void visualize_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
// {
//     pcl::visualization::CloudViewer viewer();
//     viewer.showCloud(cloud);
//     while (!viewer.wasStopped())
//     {
//         // Do nothing but wait.
//     }
// }

void calculate_narf(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Object for storing the keypoints' indices.
    pcl::PointCloud<int>::Ptr keypoints(new pcl::PointCloud<int>);
    // Object for storing the NARF descriptors.
    pcl::PointCloud<pcl::Narf36>::Ptr descriptors(new pcl::PointCloud<pcl::Narf36>);
 
    // Convert the cloud to range image.
    int imageSizeX = 640, imageSizeY = 480;
    float centerX = (640.0f / 2.0f), centerY = (480.0f / 2.0f);
    float focalLengthX = 525.0f, focalLengthY = focalLengthX;
    Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
                                 cloud->sensor_origin_[1],
                                 cloud->sensor_origin_[2])) *
                                 Eigen::Affine3f(cloud->sensor_orientation_);
    float noiseLevel = 0.0f, minimumRange = 0.0f;
    pcl::RangeImagePlanar rangeImage;
    rangeImage.createFromPointCloudWithFixedSize(*cloud, imageSizeX, imageSizeY,
            centerX, centerY, focalLengthX, focalLengthX,
            sensorPose, pcl::RangeImage::CAMERA_FRAME,
            noiseLevel, minimumRange);
 
    // Extract the keypoints.
    pcl::RangeImageBorderExtractor borderExtractor;
    pcl::NarfKeypoint detector(&borderExtractor);
    detector.setRangeImage(&rangeImage);
    detector.getParameters().support_size = 0.2f;
    detector.compute(*keypoints);
    cout << "keypoints " << *keypoints << endl;
 
    // The NARF estimator needs the indices in a vector, not a cloud.
    std::vector<int> keypoints2;
    keypoints2.resize(keypoints->points.size());
    for (unsigned int i = 0; i < keypoints->size(); ++i)
        keypoints2[i] = keypoints->points[i];
    // NARF estimation object.
    pcl::NarfDescriptor narf(&rangeImage, &keypoints2);
    // Support size: choose the same value you used for keypoint extraction.
    narf.getParameters().support_size = 0.2f;
    // If true, the rotation invariant version of NARF will be used. The histogram
    // will be shifted according to the dominant orientation to provide robustness to
    // rotations around the normal.
    narf.getParameters().rotation_invariant = true;
 
    narf.compute(*descriptors);
    cout << "descriptors " << *descriptors <<endl;
}

void calculate_pfh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Object for storing the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Object for storing the PFH descriptors for each point.
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors(new pcl::PointCloud<pcl::PFHSignature125>());
 
 
    // Note: you would usually perform downsampling now. It has been omitted here
    // for simplicity, but be aware that computation can take a long time.
    cout << "Size of point cloud before downsampling " << cloud->width * cloud->height <<endl;
    cloud = perform_downsampling(cloud);
    cout << "Size of point cloud after downsampling " << cloud->width * cloud->height <<endl;
    // visualize_pointcloud(cloud);

    // Estimate the normals.
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(0.03);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
 
    // PFH estimation object.
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    pfh.setInputCloud(cloud);
    pfh.setInputNormals(normals);
    pfh.setSearchMethod(kdtree);
    // Search radius, to look for neighbors. Note: the value given here has to be
    // larger than the radius used to estimate the normals.
    pfh.setRadiusSearch(0.05);
 
    pfh.compute(*descriptors);

    ofstream myfile;
    myfile.open ("features.csv");
    for( int i = 0; i < descriptors->points.size(); i++)
    {
    //     float tmp[size_descriptor];
    //     tmp[i] = descriptors->points[i].histogram[i];
    //     for( int j = 0; j < size_descriptor ; j++ )
    //     {
    //         myfile << descriptors->points[i].histogram[j];
    //     }
    //     myfile << endl;
    // }
        myfile << object_names[1] << ",";
        myfile << descriptors->points[i]; // << endl
    }
    myfile.close();

    // const std::string id="cloud";
    // pcl::visualization::PCLHistogramVisualizer hist; 
    // hist.addFeatureHistogram (*descriptors, 400 , id, 640, 200);

    // // while(!hist.wasStopped())
    // {    
    //     hist.spinOnce (100);
    //     boost::this_thread::sleep (boost::posix_time::microseconds (100000));  
    // }
    // char c;
    // cin>>c;

    cout << "descriptors" << *descriptors <<endl;
    cout << "descriptors " << descriptors->points[10] << endl;
    // cout << "descriptors " << descriptors->points[20] << endl;

    cout << "PFH successful !" << endl;
}

int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud = load_pcd();
    // cout << "Size of point cloud " << pointcloud->width * pointcloud->height<<endl;
    // calculate_pfh(pointcloud);
    calculate_narf(pointcloud);
    return 0;
}
