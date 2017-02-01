#include<opencv2/core.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void computeExtrisic(vector<vector<Point2f>> leftImPoints,
		vector<vector<Point2f>> rightImPoints, vector<Mat> Emat,
		Mat cameraMatrix, vector<Mat>&R_vec, vector<Mat>&t_vec);

void rescaleExtrisic(vector<Mat>&R_vec, vector<Mat>&t_vec);

void plane_sweep(vector<Mat> img, vector<Mat> R_vec, vector<Mat> t_vec,
		Mat cameraMatrix);

void testExtrisic(vector<vector<Point2f>> leftImPoints,
		vector<vector<Point2f>> rightImPoints, Mat cameraMatrix,
		vector<Mat> R_vec, vector<Mat> t_vec,vector<Mat>img);
