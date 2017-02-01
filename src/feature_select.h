#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

void surf_pt(vector<Mat> img, vector<vector<KeyPoint>> &keypoints_vec,
		vector<Mat> &descriptors_vec);

void pt_show(vector<Mat> img, vector<vector<KeyPoint>> keypoints_vec,
		bool imshow_flag);

void pt_match(vector<vector<KeyPoint>> keypoints_vec,
		vector<Mat> descriptors_vec, vector<vector<DMatch>>&good_matches_vec);

void pt_match_show(vector<Mat> img, vector<vector<KeyPoint>> keypoints_vec,
		vector<vector<DMatch>> good_matches_vec, bool imshow_flag);

void find_good_points(vector<DMatch> good_matches,
		vector<KeyPoint> keypointsLeft, vector<KeyPoint> keypointsRight,
		vector<Point2f> &selPointsLeft, vector<Point2f>&selPointsRight);

void computeEnFmat(vector<Mat> img, vector<vector<DMatch>> good_matches_vec,
		vector<vector<KeyPoint>> keypoints_vec, vector<Mat>&Emat,
		vector<Mat>&Fmat, vector<vector<Point2f>> &leftImPoints,
		vector<vector<Point2f>>&rightImPoints, Mat cameraMatrix,
		bool imshow_flag);

void applyMask(Mat mask, vector<Point2f> &selPointsLeft,
		vector<Point2f> &selPointsRight, Mat &leftImg, Mat &rightImg);

void showEpilines(Mat Fmat, Mat leftImg, Mat rightImg,
		vector<Point2f> selPointsLeft, vector<Point2f> selPointsRight);

void computeEpilines(InputArray _points, int whichImage, InputArray _Fmat,
		OutputArray _lines);
