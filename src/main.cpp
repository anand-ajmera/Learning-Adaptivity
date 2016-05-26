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
    #include <pcl/io/pcd_io.h>
    #include <pcl/keypoints/iss_3d.h>
    #include <pcl/features/fpfh.h>

    using namespace std;

    //int size_descriptor = 125;
    const int sample_size = 53;
    const int object_count = 4;
    const int sample_keypoint_size = 10;
    const string data_folder = "../at_home/";
    const string target_folder = "../features/";
    string object_names[4] = {"MuscleBox","BigCoffeeCup","Pringles","SmallKetchupBottle"};
    string str;



     
    // This function by Tommaso Cavallari and Federico Tombari, taken from the tutorial
    // http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
    double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        double resolution = 0.0;
        int numberOfPoints = 0;
        int nres;
        std::vector<int> indices(2);
        std::vector<float> squaredDistances(2);
        pcl::search::KdTree<pcl::PointXYZ> tree;
        tree.setInputCloud(cloud);
     
        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (! pcl_isfinite((*cloud)[i].x))
                continue;
     
            // Considering the second neighbor since the first is the point itself.
            nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
            if (nres == 2)
            {
                resolution += sqrt(squaredDistances[1]);
                ++numberOfPoints;
            }
        }
        if (numberOfPoints != 0)
            resolution /= numberOfPoints;
     
        return resolution;
    }

    void load_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,string str)
    {  
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

        //str.append(data_folder);
        //str.append(object_names[1]);
        ///str.append("/5.pcd");

        if (pcl::io::loadPCDFile<pcl::PointXYZ> (str, *cloud) == -1) //* load the file
        {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        // return (-1);
        }
        //std::cout << "Loaded ";
        //return cloud;
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

    void calculate_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints)
    {
        
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
        detector.setInputCloud(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        detector.setSearchMethod(kdtree);
        double resolution = computeCloudResolution(cloud);
        // Set the radius of the spherical neighborhood used to compute the scatter matrix.
        detector.setSalientRadius(6 * resolution);
        // Set the radius for the application of the non maxima supression algorithm.
        detector.setNonMaxRadius(4 * resolution);
        // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
        detector.setMinNeighbors(5);
        // Set the upper bound on the ratio between the second and the first eigenvalue.
        detector.setThreshold21(0.975);
        // Set the upper bound on the ratio between the third and the second eigenvalue.
        detector.setThreshold32(0.975);
        // Set the number of prpcessing threads to use. 0 sets it to automatic.
        detector.setNumberOfThreads(4);
     
        detector.compute(*keypoints);
        //cout<<"Keypoints"<<*keypoints<<endl;
        
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

    void calculate_fpfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,
                                pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors)
    {
            // Object for storing the normals.
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // Object for storing the FPFH descriptors for each point.
        //pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

            // Estimate the normals.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(cloud);
        normalEstimation.setRadiusSearch(0.03);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.compute(*normals);
     
        // FPFH estimation object.
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setSearchSurface(cloud);
        fpfh.setInputCloud(keypoints);
        fpfh.setInputNormals(normals);
        fpfh.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        fpfh.setRadiusSearch(0.05);
     
        fpfh.compute(*descriptors);
        //cout<<"descriptors"<<*descriptors<<endl;

    }

    void sample_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_keypoints)
    {

    }

    void visualize_histogram(pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors)
    {
        const std::string id="cloud";
        pcl::visualization::PCLHistogramVisualizer hist; 
        hist.addFeatureHistogram (*descriptors, 33 , id, 640, 200);

        // while(!hist.wasStopped())
        {    
            hist.spin();
            //boost::this_thread::sleep (boost::posix_time::microseconds (100000));  
        }
    }

    void write_to_file(pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors,const char* filename,const int& idx)
    {
        //pcl::io::savePCDFile (filename, *descriptors);
        ofstream myfile;
        myfile.open (filename);
        //myfile << object_names[idx] <<endl;
        for( int i = 0; i < descriptors->points.size(); i++)
            {
                if(i < sample_keypoint_size)
                {
                    for(int j=0;j<33;j++)
                    {
                        myfile << descriptors->points[i].histogram[j];
                        if(j!=32)
                        myfile<<",";  
                    }
                
                    myfile<< endl;    
                }
            
        }
        myfile.close();
        cout<<"saved to file"<<filename<<endl;
    }

    int main (int argc, char** argv)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
        // cout << "Size of point cloud " << pointcloud->width * pointcloud->height<<endl;
        // calculate_pfh(pointcloud);
        //string in_file(data_folder),out_file(target_folder);
        for(int i = 0;i < object_count; i++)
        {
            //in_file.append(object_names[i]);
            for(int j=1;j<= sample_size;j++)
            {   string in_file(data_folder);
                in_file.append(object_names[i]+"/");
                in_file.append(boost::to_string(j)+".pcd");
                //cout<<in_file<<endl;

                //load pcd
                load_pcd(cloud,in_file);
                if(cloud != 0)
                {
                    //cout<<"point cloud loaded"<<endl;
                        calculate_keypoints(cloud,keypoints);
                        calculate_fpfh_feature(cloud,keypoints,descriptors);
                        //cout<<"descriptors"<<descriptors->points[0]<<endl;
                        //saving the features to pcd file
                        string out_file(target_folder);
                        out_file.append(object_names[i]+"/");
                        out_file.append(boost::to_string(j)+".csv");  
                        write_to_file(descriptors,out_file.c_str(),i);
                        //visualize_histogram(descriptors);
                        //cout<<"object written"<<endl;
                }
                else
                {
                    cerr<<"Cloud pointer is NULL"<<endl;
                }


            }
        }
            return 0;
    }
