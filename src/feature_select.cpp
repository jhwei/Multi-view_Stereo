#include"feature_select.h"
#include"opencv2/calib3d.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/imgproc.hpp"
#include<iostream>
void surf_pt(vector<Mat> img, vector<vector<KeyPoint>> &keypoints_vec,
		vector<Mat> &descriptors_vec)
{
	//Ptr<xfeatures2d::SIFT> sift_ptr = xfeatures2d::SIFT::create();
	Ptr<xfeatures2d::SURF> surf_ptr = xfeatures2d::SURF::create(400, true);
	for (unsigned int i = 0; i < img.size(); i++)
	{
		vector<KeyPoint> key_points;
		Mat descriptors;
		//sift_ptr->detectAndCompute(img[i], noArray(), key_points, descriptors);
		surf_ptr->detectAndCompute(img[i], noArray(), key_points, descriptors);
		keypoints_vec.push_back(key_points);
		descriptors_vec.push_back(descriptors);
	}
}

void pt_show(vector<Mat> img, vector<vector<KeyPoint>> keypoints_vec,
		bool imshow_flag)
{
	if (!imshow_flag)
		return;
	vector<Mat> img_keypoints;
	for (unsigned int i = 0; i < img.size(); i++)
	{
		Mat img_tmp;

		drawKeypoints(img[i], keypoints_vec[i], img_tmp, Scalar::all(-1),
				DrawMatchesFlags::DEFAULT);
		img_keypoints.push_back(img_tmp);
	}

	imshow("img1", img_keypoints[0]);
	imshow("img2", img_keypoints[1]);
	imshow("img3", img_keypoints[2]);
	cout << "\nFeature points detected by surf are shown now" << endl;
	cout << "Press any key to continue.\n" << endl;
	waitKey(0);
}

void pt_match(vector<vector<KeyPoint>> keypoints_vec,
		vector<Mat> descriptors_vec, vector<vector<DMatch>>&good_matches_vec)
{
	for (unsigned int i = 0; i < keypoints_vec.size(); i++)
	{
		Mat descriptors1, descriptors2;
		vector<KeyPoint> keypoints1, keypoints2;

		if (i == keypoints_vec.size() - 1)
		{
			descriptors_vec[0].copyTo(descriptors1);
			keypoints1 = keypoints_vec[0];
			descriptors_vec[i].copyTo(descriptors2);
			keypoints2 = keypoints_vec[i];
		}
		else
		{
			descriptors_vec[i].copyTo(descriptors1);
			keypoints1 = keypoints_vec[i];
			descriptors_vec[i + 1].copyTo(descriptors2);
			keypoints2 = keypoints_vec[i + 1];
		}

		FlannBasedMatcher Flann_matcher; // FLANN - Fast Library for Approximate Nearest Neighbors
		vector<vector<DMatch> > matches;

		Flann_matcher.knnMatch(descriptors1, descriptors2, matches, 2); // find the best 2 matches of each descriptor

		vector<DMatch> good_matches;

		for (int k = 0; k < min(descriptors2.rows - 1, (int) matches.size());
				k++)
		{
			if ((matches[k][0].distance < 0.25 * (matches[k][1].distance))
					&& ((int) matches[k].size() <= 2
							&& (int) matches[k].size() > 0))

				good_matches.push_back(matches[k][0]);

		}

		good_matches_vec.push_back(good_matches);
	}
}

void pt_match_show(vector<Mat> img, vector<vector<KeyPoint>> keypoints_vec,
		vector<vector<DMatch>> good_matches_vec, bool imshow_flag)
{
	if (!imshow_flag)
		return;

	vector<Mat> img_matches;
	for (unsigned int i = 0; i < img.size(); i++)
	{
		Mat leftImg, rightImg;
		vector<Point2f> selPointsLeft, selPointsRight;
		vector<KeyPoint> keypointsLeft, keypointsRight;
		if (i == img.size() - 1)
		{
			keypointsLeft = keypoints_vec[0];
			img[0].copyTo(leftImg);
			keypointsRight = keypoints_vec[i];
			img[i].copyTo(rightImg);
		}
		else
		{
			keypointsLeft = keypoints_vec[i];
			img[i].copyTo(leftImg);
			keypointsRight = keypoints_vec[i + 1];
			img[i + 1].copyTo(rightImg);
		}

		find_good_points(good_matches_vec[i], keypointsLeft, keypointsRight,
				selPointsLeft, selPointsRight);

		Mat img_match;
		drawMatches(leftImg, keypointsLeft, rightImg, keypointsRight,
				good_matches_vec[i], img_match, Scalar::all(-1),
				Scalar::all(-1), vector<char>(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		img_matches.push_back(img_match);
	}

	imshow("img1", img_matches[0]);
	imshow("img2", img_matches[1]);
	imshow("img3", img_matches[2]);
	cout << "\nFeature point match are shown in each image pair right now."
			<< endl;
	cout << "Press any key to continue.\n" << endl;
	waitKey(0);
}

void find_good_points(vector<DMatch> good_matches,
		vector<KeyPoint> keypointsLeft, vector<KeyPoint> keypointsRight,
		vector<Point2f> &selPointsLeft, vector<Point2f>&selPointsRight)
{
	vector<int> pointIndexesLeft, pointIndexesRight;
	for (std::vector<cv::DMatch>::const_iterator it = good_matches.begin();
			it != good_matches.end(); ++it)
	{
		pointIndexesLeft.push_back(it->queryIdx);
		pointIndexesRight.push_back(it->trainIdx);
	}

	KeyPoint::convert(keypointsLeft, selPointsLeft, pointIndexesLeft);
	KeyPoint::convert(keypointsRight, selPointsRight, pointIndexesRight);
}

void computeEnFmat(vector<Mat> img, vector<vector<DMatch>> good_matches_vec,
		vector<vector<KeyPoint>> keypoints_vec, vector<Mat>&Emat,
		vector<Mat>&Fmat, vector<vector<Point2f>> &leftImPoints,
		vector<vector<Point2f>>&rightImPoints, Mat cameraMatrix,
		bool imshow_flag)
{
	for (unsigned int i = 0; i < good_matches_vec.size(); i++)
	{
		vector<Point2f> selPointsLeft, selPointsRight;
		vector<KeyPoint> keypointsLeft, keypointsRight;
		Mat leftImg, rightImg;
		if (i == good_matches_vec.size() - 1)
		{
			keypointsLeft = keypoints_vec[0];
			keypointsRight = keypoints_vec[i];
			img[0].copyTo(leftImg);
			img[i].copyTo(rightImg);
		}
		else
		{
			keypointsLeft = keypoints_vec[i];
			keypointsRight = keypoints_vec[i + 1];
			img[i].copyTo(leftImg);
			img[i + 1].copyTo(rightImg);
		}

		find_good_points(good_matches_vec[i], keypointsLeft, keypointsRight,
				selPointsLeft, selPointsRight);
		Mat F_tmp, E_tmp;
		Mat mask;
		F_tmp = findFundamentalMat(selPointsLeft, selPointsRight, FM_RANSAC, 1,
				0.99, mask);

		applyMask(mask, selPointsLeft, selPointsRight, leftImg, rightImg);
		E_tmp = findEssentialMat(selPointsLeft, selPointsRight, cameraMatrix);

		Fmat.push_back(F_tmp);
		Emat.push_back(E_tmp);

		cout << "Essential Matrix for image pair " << i << " is: " << endl;
		cout << E_tmp << endl;

		showEpilines(F_tmp, leftImg, rightImg, selPointsLeft, selPointsRight);
		leftImPoints.push_back(selPointsLeft);
		rightImPoints.push_back(selPointsRight);
	}
}

void applyMask(Mat mask, vector<Point2f> &selPointsLeft,
		vector<Point2f> &selPointsRight, Mat &leftImg, Mat &rightImg)
{
	vector<Point2f> realPointsLeft, realPointsRight;

	for (int i = 0; i < mask.size().height; i++)
	{
		int flag = mask.at<uchar>(i, 0);

		if (flag == 1)
		{
			realPointsLeft.push_back(selPointsLeft[i]);
			realPointsRight.push_back(selPointsRight[i]);

			circle(rightImg, selPointsRight[i], 10, Scalar(0, 255, 255), 5);
			circle(leftImg, selPointsLeft[i], 10, Scalar(0, 255, 255), 5);

		}
		else
		{
			circle(leftImg, selPointsLeft[i], 15, Scalar(0, 0, 0), 5);
			circle(rightImg, selPointsRight[i], 15, Scalar(0, 0, 0), 5);
		}
		selPointsLeft = realPointsLeft;
		selPointsRight = realPointsRight;
	}
}

void showEpilines(Mat Fmat, Mat leftImg, Mat rightImg,
		vector<Point2f> selPointsLeft, vector<Point2f> selPointsRight)
{
	vector<Vec3f> linesLeft;
	computeEpilines(Mat(selPointsLeft), 1, Fmat, linesLeft);

	vector<cv::Vec3f> linesRight;
	computeEpilines(Mat(selPointsRight), 2, Fmat, linesRight);

	vector<cv::Vec3f>::const_iterator itl = linesLeft.begin();
	vector<cv::Vec3f>::const_iterator itr = linesRight.begin();
	RNG rng(0);

	while (itl != linesLeft.end() && itr != linesRight.end())
	{
		int icolor = (unsigned) rng;
		Scalar color = Scalar(icolor & 255, (icolor >> 8) & 255,
				(icolor >> 16) & 255);

		line(rightImg, Point(0, -(*itl)[2] / (*itl)[1]),
				Point(rightImg.cols,
						-((*itl)[2] + (*itl)[0] * rightImg.cols) / (*itl)[1]),
				color, 3);

		line(leftImg, Point(0, -(*itr)[2] / (*itr)[1]),
				Point(leftImg.cols,
						-((*itr)[2] + (*itr)[0] * leftImg.cols) / (*itr)[1]),
				color, 3);
		itl++;
		itr++;
	}

	cv::imshow("img2", rightImg);
	cv::imshow("img1", leftImg);
	cout
			<< "\nEpipolar lines together with inlier and outlier points for image pair are shown in img1 and img2."
			<< endl;
	cout << "Press any key to continue.\n" << endl;
	waitKey(0);
}

void computeEpilines(InputArray _points, int whichImage, InputArray _Fmat,
		OutputArray _lines)
{
	double f[9];
	Mat tempF(3, 3, CV_64F, f);
	Mat points = _points.getMat(), F = _Fmat.getMat();

	if (!points.isContinuous())
		points = points.clone();

	int npoints = points.checkVector(2);
	if (npoints < 0)
	{
		npoints = points.checkVector(3);
		if (npoints < 0)
			CV_Error(Error::StsBadArg,
					"The input should be a 2D or 3D point set");
		Mat temp;
		convertPointsFromHomogeneous(points, temp);
		points = temp;
	}
	int depth = points.depth();

	F.convertTo(tempF, CV_64F);
	if (whichImage == 2)
		transpose(tempF, tempF);

	int ltype = CV_MAKETYPE(MAX(depth, CV_32F), 3);
	_lines.create(npoints, 1, ltype);
	Mat lines = _lines.getMat();
	if (!lines.isContinuous())
	{
		_lines.release();
		_lines.create(npoints, 1, ltype);
		lines = _lines.getMat();
	}

	if (depth == CV_32S || depth == CV_32F)
	{
		const Point* ptsi = points.ptr<Point>();
		const Point2f* ptsf = points.ptr<Point2f>();
		Point3f* dstf = lines.ptr<Point3f>();
		for (int i = 0; i < npoints; i++)
		{
			Point2f pt =
					depth == CV_32F ?
							ptsf[i] :
							Point2f((float) ptsi[i].x, (float) ptsi[i].y);
			double a = f[0] * pt.x + f[1] * pt.y + f[2];
			double b = f[3] * pt.x + f[4] * pt.y + f[5];
			double c = f[6] * pt.x + f[7] * pt.y + f[8];
			double nu = a * a + b * b;
			nu = nu ? 1. / std::sqrt(nu) : 1.;
			a *= nu;
			b *= nu;
			c *= nu;
			dstf[i] = Point3f((float) a, (float) b, (float) c);
		}
	}
	else
	{
		const Point2d* ptsd = points.ptr<Point2d>();
		Point3d* dstd = lines.ptr<Point3d>();
		for (int i = 0; i < npoints; i++)
		{
			Point2d pt = ptsd[i];
			double a = f[0] * pt.x + f[1] * pt.y + f[2];
			double b = f[3] * pt.x + f[4] * pt.y + f[5];
			double c = f[6] * pt.x + f[7] * pt.y + f[8];
			double nu = a * a + b * b;
			nu = nu ? 1. / std::sqrt(nu) : 1.;
			a *= nu;
			b *= nu;
			c *= nu;
			dstd[i] = Point3d(a, b, c);
		}
	}
}

