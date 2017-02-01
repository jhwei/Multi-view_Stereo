#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <string.h>
#include<vector>

#include"calibration.h"
#include"feature_select.h"
#include"depth_cal.h"

int main()
{
	cout << "Program started."<<endl;
	bool imshow_flag = true;
	string calib_img_filename = "./data/calibration/calib";
	string calib_data_filename = "./data/calibration/calib_param.yml";
	Mat cameraMatrix;
	if (false)
	{
		cout << "\nStart calibration." << endl;
		cameraCalibration(calib_img_filename, calib_data_filename);
		destroyAllWindows();
		cout << "\nCalibration complete. Parameters are stored in "
				<< calib_data_filename << endl;
		//return 0;
	}

	vector<Mat> img_pre, img;
	string img_seq_name = "data/image_sequence.yml";
	readImageList(img_seq_name, img_pre);

	namedWindow("img1", CV_WINDOW_NORMAL);
	resizeWindow("img1", 1024, 768);
	namedWindow("img2", CV_WINDOW_NORMAL);
	resizeWindow("img2", 1024, 768);
	namedWindow("img3", CV_WINDOW_NORMAL);
	resizeWindow("img3", 1024, 768);

	img_undistort(calib_data_filename, img_pre, img, cameraMatrix, imshow_flag);

	vector<vector<KeyPoint>> keypoints_vec;
	vector<Mat> descriptors_vec;

	surf_pt(img, keypoints_vec, descriptors_vec);
	pt_show(img, keypoints_vec, imshow_flag);

	vector<vector<DMatch>> good_matches_vec;

	pt_match(keypoints_vec, descriptors_vec, good_matches_vec);
	pt_match_show(img, keypoints_vec, good_matches_vec, imshow_flag);

	vector<vector<Point2f>> selpoints;

	vector<Mat> Emat, Fmat;
	vector<Mat> R_vec, t_vec;
	vector<vector<Point2f>> leftImPoints;
	vector<vector<Point2f>> rightImPoints;

	computeEnFmat(img, good_matches_vec, keypoints_vec, Emat, Fmat,
			leftImPoints, rightImPoints, cameraMatrix, imshow_flag);
	computeExtrisic(leftImPoints, rightImPoints, Emat, cameraMatrix, R_vec,
			t_vec);
	testExtrisic(leftImPoints, rightImPoints, cameraMatrix, R_vec, t_vec, img);
	rescaleExtrisic(R_vec, t_vec);
	plane_sweep(img, R_vec, t_vec, cameraMatrix);
	return 0;
}
