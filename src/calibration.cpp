#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include"calibration.h"

enum
{
	DETECTION = 0, CAPTURING = 1, CALIBRATED = 2
};
enum Pattern
{
	CHESSBOARD, CIRCLES_GRID
};

void readCameraParams(string &filename, Mat& cameraMatrix, Size &imageSize,
		Mat& distCoeffs)
{
	FileStorage fs(filename, FileStorage::READ);

	fs["image_width"] >> imageSize.width;
	fs["image_height"] >> imageSize.height;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
}

double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints, const vector<Mat>& rvecs,
		const vector<Mat>& tvecs, const Mat& cameraMatrix,
		const Mat& distCoeffs, vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int) objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
				distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int) objectPoints[i].size();
		perViewErrors[i] = (float) std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

void calcChessboardCorners(Size boardSize, float squareSize,
		vector<Point3f>& corners)
{
	Pattern patternType = CHESSBOARD;
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(
						Point3f(float(j * squareSize), float(i * squareSize),
								0));
		break;

	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

bool runCalibration(vector<vector<Point2f> > imagePoints, Size imageSize,
		Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<Mat>& rvecs, vector<Mat>& tvecs, vector<float>& reprojErrs,
		double& totalAvgErr)
{
	cameraMatrix = Mat::eye(3, 3, CV_64F);

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0]);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms = calibrateCamera(objectPoints, imagePoints, imageSize,
			cameraMatrix, distCoeffs, rvecs, tvecs,
			0 | CALIB_FIX_K4 | CALIB_FIX_K5);
	printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs,
			tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

void saveCameraParams(const string& filename, Size imageSize, Size boardSize,
		float squareSize, const Mat& cameraMatrix, const Mat& distCoeffs,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const vector<float>& reprojErrs,
		const vector<vector<Point2f> >& imagePoints, double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int) std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;
	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int) rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int) rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		//cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int) imagePoints.size(), (int) imagePoints[0].size(),
		CV_32FC2);
		for (int i = 0; i < (int) imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

void readImageList(const string& filename, vector<Mat>& img)
{
	FileStorage fs(filename, FileStorage::READ);
	FileNode n = fs.getFirstTopLevelNode();

	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		Mat img_tmp;
		img_tmp = imread((string) *it);
		img.push_back(img_tmp);
	}
}

bool readStringList(const string& filename, vector<string>& l, Size &boardSize,
		float &squareSize)
{
	l.resize(0);

	FileStorage fs1(filename + "_image.yml", FileStorage::READ);
	if (!fs1.isOpened())
		return false;

	FileNode n = fs1.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string) *it);

	FileStorage fs2(filename + "_chessboard.yml", FileStorage::READ);

	fs2["board_width"] >> boardSize.width;
	fs2["board_height"] >> boardSize.height;
	fs2["square_size"] >> squareSize;

	return true;
}

bool runAndSave(const string& outputFilename,
		const vector<vector<Point2f> >& imagePoints, Size imageSize,
		Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs,
		bool writeExtrinsics, bool writePoints)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize,
			cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, totalAvgErr);
	printf("%s. avg reprojection error = %.2f\n",
			ok ? "Calibration succeeded" : "Calibration failed", totalAvgErr);

	if (ok)
		saveCameraParams(outputFilename, imageSize, boardSize, squareSize,
				cameraMatrix, distCoeffs,
				writeExtrinsics ? rvecs : vector<Mat>(),
				writeExtrinsics ? tvecs : vector<Mat>(),
				writeExtrinsics ? reprojErrs : vector<float>(),
				writePoints ? imagePoints : vector<vector<Point2f> >(),
				totalAvgErr);
	return ok;
}

void cameraCalibration(string &inputFilename, string &outputFilename)
{
	Size boardSize, imageSize;
	float squareSize = 29;
	Mat cameraMatrix, distCoeffs;

	int i, nframes;
	bool writeExtrinsics = true;
	bool writePoints = true;

	clock_t prevTimestamp = 0;
	int mode = DETECTION;

	vector<vector<Point2f> > imagePoints;
	vector<string> imageList;

	if (!inputFilename.empty()
			&& readStringList(inputFilename, imageList, boardSize, squareSize))
		mode = CAPTURING;

	if (imageList.empty())
	{
		cerr << "Could not initialize video (%d) capture\n" << endl;
		exit(-1);
	}

	if (!imageList.empty())
		nframes = (int) imageList.size();

	if (nframes < 4)
	{
		cerr << "not enough images to calibrate." << endl;
		exit(-1);
	}

	namedWindow("Image View", CV_WINDOW_NORMAL);
	resizeWindow("Image View", 1024, 768);

	for (i = 0;; i++)
	{
		Mat view, viewGray;

		if (i < (int) imageList.size())
			view = imread(imageList[i], 1);

		if (view.empty())
		{
			if (i < (int) imageList.size())
			{
				cerr << "invalid image name" << imageList[i] << endl;
				exit(-1);
			}
			if (imagePoints.size() > 0)
				runAndSave(outputFilename, imagePoints, imageSize, boardSize,
						squareSize, cameraMatrix, distCoeffs, writeExtrinsics,
						writePoints);
			break;
		}

		imageSize = view.size();

		vector<Point2f> pointbuf;
		cvtColor(view, viewGray, COLOR_BGR2GRAY);

		bool found;

		found = findChessboardCorners(view, boardSize, pointbuf,
				CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK
						| CALIB_CB_NORMALIZE_IMAGE);

		// improve the found corners' coordinate accuracy
		if (found)
			cornerSubPix(viewGray, pointbuf, Size(11, 11), Size(-1, -1),
					TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30,
							0.1));

		if (mode == CAPTURING && found
				&& (clock() - prevTimestamp > 1 * 1e-3 * CLOCKS_PER_SEC))
		{
			imagePoints.push_back(pointbuf);
			prevTimestamp = clock();
		}

		if (found)
			drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

		string msg =
				mode == CAPTURING ? "100/100" :
				mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10,
				view.rows - 2 * baseLine - 10);

		if (mode == CAPTURING)
			msg = format("%d/%d", (int) imagePoints.size(), nframes);

		putText(view, msg, textOrigin, 1, 1,
				mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

		if (mode == CALIBRATED)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}

		imshow("Image View", view);
		int key = 0xff & waitKey(200);

		if ((key & 255) == 27)
			break;

		if (mode == CAPTURING && imagePoints.size() >= (unsigned) nframes)
		{
			if (runAndSave(outputFilename, imagePoints, imageSize, boardSize,
					squareSize, cameraMatrix, distCoeffs, writeExtrinsics,
					writePoints))
				mode = CALIBRATED;
			else
				mode = DETECTION;

			break;
		}
	}
}

void img_undistort(string &filename, vector<Mat> img_pre, vector<Mat>&img,
		Mat&cameraMatrix, bool imshow_flag)
{
	Mat distCoeffs, map1, map2;
	Size imageSize;

	readCameraParams(filename, cameraMatrix, imageSize, distCoeffs);
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
			getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1,
					imageSize, 0), imageSize, CV_16SC2, map1, map2);

	for (unsigned int i = 0; i < img_pre.size(); i++)
	{
		Mat img_tmp;
		undistort(img_pre[i], img_tmp, cameraMatrix, distCoeffs, cameraMatrix);
		remap(img_pre[i], img_tmp, map1, map2, INTER_CUBIC);
		img.push_back(img_tmp);
	}
	imshow("img1", img[0]);
	imshow("img2", img[1]);
	imshow("img3", img[2]);

	waitKey(10);
}
