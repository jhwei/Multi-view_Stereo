#include<opencv2/core.hpp>
#include<iostream>

using namespace cv;
using namespace std;

double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints, const vector<Mat>& rvecs,
		const vector<Mat>& tvecs, const Mat& cameraMatrix,
		const Mat& distCoeffs, vector<float>& perViewErrors);

void calcChessboardCorners(Size boardSize, float squareSize,
		vector<Point3f>& corners);

bool runCalibration(vector<vector<Point2f> > imagePoints, Size imageSize,
		Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<Mat>& rvecs, vector<Mat>& tvecs, vector<float>& reprojErrs,
		double& totalAvgErr);

void saveCameraParams(const string& filename, Size imageSize, Size boardSize,
		float squareSize, const Mat& cameraMatrix, const Mat& distCoeffs,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const vector<float>& reprojErrs,
		const vector<vector<Point2f> >& imagePoints, double totalAvgErr);

bool readStringList(const string& filename, vector<string>& l, Size &boardSize,
		float &squareSize);

bool runAndSave(const string& outputFilename,
		const vector<vector<Point2f> >& imagePoints, Size imageSize,
		Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs,
		bool writeExtrinsics, bool writePoints);

void cameraCalibration(string &inputFilename, string &outputFilename);

void readCameraParams(string &filename, Mat& cameraMatrix, Size &imageSize,
		Mat& distCoeffs);
void readImageList(const string& filename, vector<Mat> &img);

void img_undistort(string &filename, vector<Mat> img_pre, vector<Mat>&img,
		Mat&cameraMatrix, bool imshow_flag);
