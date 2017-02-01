#include"depth_cal.h"
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>

void computeExtrisic(vector<vector<Point2f>> leftImPoints,
		vector<vector<Point2f>> rightImPoints, vector<Mat> Emat,
		Mat cameraMatrix, vector<Mat>&R_vec, vector<Mat>&t_vec)
{
	for (unsigned int i = 0; i < Emat.size(); i++)
	{
		Mat R, t;
		SVD svd_E = SVD(Emat[i]);
		Mat W = Mat::zeros(3, 3, CV_64F);
		W.at<double>(1, 0) = 1;
		W.at<double>(0, 1) = -1;
		W.at<double>(2, 2) = 1;
		Mat R1 = svd_E.u * W * svd_E.vt;
		Mat R2 = svd_E.u * W.t() * svd_E.vt;
		Mat t1 = svd_E.u.col(2);
		Mat t2 = -1.0 * t1;
		recoverPose(Emat[i], leftImPoints[i], rightImPoints[i], cameraMatrix, R,
				t);
		R_vec.push_back(R);
		t_vec.push_back(t);
		cout << "\nRotation matrix for image pair " << i << endl;
		cout << R << endl;
		cout << "\nTranslation vector for image pair " << i << endl;
		cout << t << endl;
	}

}
void rescaleExtrisic(vector<Mat>&R_vec, vector<Mat>&t_vec)
{
	Mat t1 = R_vec[0].t() * t_vec[0];
	Mat t2 = R_vec[2].t() * t_vec[1];
	Mat t3 = -R_vec[2].t() * t_vec[2];

	cout << "||t1+t2+t3||: " << (t1 + t2 + t3).t() * (t1 + t2 + t3) << endl;

	Mat t = Mat::zeros(3, 3, CV_64F);
	t1.copyTo(t.col(0));
	t2.copyTo(t.col(1));
	t3.copyTo(t.col(2));
	SVD svd = SVD(t);
	svd.vt = svd.vt.t();

	double beta = 1;
	double gamma = 1;
	beta = svd.vt.at<double>(1, 2) / svd.vt.at<double>(0, 2);
	gamma = svd.vt.at<double>(2, 2) / svd.vt.at<double>(0, 2);
	cout << "New ||t1+beta t2+gamma t3||: "
			<< (t1 + t2 * beta + t3 * gamma).t() * (t1 + t2 * beta + t3 * gamma)
			<< endl;
	t_vec[1] = t_vec[1] * beta;
	t_vec[2] = t_vec[2] * gamma;
	cout << "beta: " << beta << ". gamma: " << gamma << "\n" << endl;
}

void testExtrisic(vector<vector<Point2f>> leftImPoints,
		vector<vector<Point2f>> rightImPoints, Mat cameraMatrix,
		vector<Mat> R_vec, vector<Mat> t_vec, vector<Mat> img)
{
	double f = (cameraMatrix.at<double>(0, 0) + cameraMatrix.at<double>(0, 0))
			/ 2;
	Mat M = Mat::eye(3, 3, CV_64F);
	M.at<double>(0, 2) = -cameraMatrix.at<double>(0, 2);
	M.at<double>(1, 2) = -cameraMatrix.at<double>(1, 2);
	M.at<double>(2, 2) = f;
	for (unsigned int i = 0; i < R_vec.size(); i++)
	{
		InputArray lpoints = leftImPoints[i];
		InputArray rpoints = rightImPoints[i];

		Mat p_l = Mat::zeros(leftImPoints[i].size(), 3, CV_64F);
		Mat x_r = Mat::zeros(leftImPoints[i].size(), 3, CV_64F);
		for (unsigned int m = 0; m < leftImPoints[i].size(); m++)
		{
			Mat lpoint = Mat::ones(3, 1, CV_64F);
			Mat rpoint = Mat::ones(3, 1, CV_64F);
			lpoint.at<double>(0, 0) = leftImPoints[i][m].x;
			lpoint.at<double>(1, 0) = leftImPoints[i][m].y;
			rpoint.at<double>(0, 0) = rightImPoints[i][m].x;
			rpoint.at<double>(1, 0) = rightImPoints[i][m].y;
			lpoint = M * lpoint;
			rpoint = M * rpoint;
			Mat z;

			z = f
					* (f * R_vec[i].row(0)
							- rpoint.at<double>(0, 0) * R_vec[i].row(2))
					* (-R_vec[i].inv() * t_vec[i])
					/ ((f * R_vec[i].row(0)
							- rpoint.at<double>(0, 0) * R_vec[i].row(2))
							* lpoint);
			double pz = z.at<double>(0, 0);
			p_l.at<double>(m, 2) = pz;
			p_l.at<double>(m, 0) = lpoint.at<double>(0, 0) * pz / f;
			p_l.at<double>(m, 1) = lpoint.at<double>(1, 0) * pz / f;
			x_r.at<double>(m, 0) = rpoint.at<double>(0, 0);
			x_r.at<double>(m, 1) = rpoint.at<double>(1, 0);
			x_r.at<double>(m, 2) = rpoint.at<double>(2, 0);
		}
		Mat p_r;
		p_r = (R_vec[i] * p_l.t()).t();
		p_r.col(0) = f * (p_r.col(0) + t_vec[i].at<double>(0, 0))
				/ (p_r.col(2) + t_vec[i].at<double>(2, 0));
		p_r.col(1) = f * (p_r.col(1) + t_vec[i].at<double>(1, 0))
				/ (p_r.col(2) + t_vec[i].at<double>(2, 0));
		p_r.col(2) = f * (p_r.col(2) + t_vec[i].at<double>(2, 0))
				/ (p_r.col(2) + t_vec[i].at<double>(2, 0));
		//cout << p_l << endl;
		p_r = (M.inv() * p_r.t()).t();
		x_r = (M.inv() * x_r.t()).t();
		double repro_error = norm(p_r, x_r, NORM_L2)
				/ (double) p_r.size().height;

		cout << "Reprojection error for image pair " << i << " is:"
				<< repro_error << endl;

		Mat showim;
		if (i == R_vec.size() - 1)
			img[2].copyTo(showim);
		else
			img[i + 1].copyTo(showim);
		for (int num = 0; num < p_r.size().height; num++)
		{
			circle(showim,
					Point(x_r.at<double>(num, 0), x_r.at<double>(num, 1)), 10,
					Scalar(0, 255, 255), 5);
			circle(showim,
					Point(p_r.at<double>(num, 0), p_r.at<double>(num, 1)), 15,
					Scalar(0, 0, 0), 5);
		}
		imshow("img2", showim);

		cout << "\nimg2 shows original points and reprojected points.\n"
				<< "Original points marked with yellow and smaller radius.\n"
				<< "Reprojected points marked with black with larger radius."
				<< endl;
		cout << "press any key to continue to the next image.\n" << endl;
		waitKey(0);
	}
}

void plane_sweep(vector<Mat> img, vector<Mat> R_vec, vector<Mat> t_vec,
		Mat cameraMatrix)
{
	int max_depth = (cameraMatrix.at<double>(0, 0)
			+ cameraMatrix.at<double>(1, 1)) / 2;
	int saturated_depth = 15;

	Size img_size = img[0].size();
	int delta = 10;
	Mat n_t = Mat::zeros(1, 3, CV_64FC1);
	n_t.at<double>(0, 2) = -1;
	Mat kernel = Mat::ones((2 * delta + 1), (delta * 2 + 1), CV_8UC1);

	Mat sad12 = Mat::ones(img_size.height, img_size.width, CV_64FC1) * 255
			* ((2 * delta + 1) * (2 * delta + 1));
	Mat sad13 = Mat::ones(img_size.height, img_size.width, CV_64FC1) * 255
			* ((2 * delta + 1) * (2 * delta + 1));
	Mat sad3 = Mat::ones(img_size.height, img_size.width, CV_64FC1) * 255
			* ((2 * delta + 1) * (2 * delta + 1));

	Mat depth12 = Mat::zeros(img_size.height, img_size.width, CV_64FC1);
	Mat depth13 = Mat::zeros(img_size.height, img_size.width, CV_64FC1);
	Mat depth3 = Mat::zeros(img_size.height, img_size.width, CV_64FC1);

	Mat gray1, gray2, gray3;
	cvtColor(img[0], gray1, CV_BGR2GRAY);
	cvtColor(img[1], gray2, CV_BGR2GRAY);
	cvtColor(img[2], gray3, CV_BGR2GRAY);

	cout << "\n--start plane-sweeping.--    (It will take a long time...)\n"
			<< endl;

	for (int i = max_depth; i > 0; i--)
	{
		float d = float(max_depth) / i;
		if (d > saturated_depth)
			break;

		Mat H12 = R_vec[0] - t_vec[0] * n_t / d;
		Mat H13 = R_vec[2] - t_vec[2] * n_t / d;

		Mat khk12 = cameraMatrix * H12 * cameraMatrix.inv();
		Mat khk13 = cameraMatrix * H13 * cameraMatrix.inv();

		Mat warp2, warp3;
		warpPerspective(gray2, warp2, khk12.inv(), img_size);
		warpPerspective(gray3, warp3, khk13.inv(), img_size);

		Mat diff12, diff13, diff3;

		filter2D(abs(gray1 - warp2), diff12, CV_64F, kernel);
		filter2D(abs(gray1 - warp3), diff13, CV_64F, kernel);
		diff3 = diff12 + diff13;

		for (int row = 0; row < img_size.height; row++)
		{
			for (int col = 0; col < img_size.width; col++)
			{
				int tmp;
				tmp = diff12.at<double>(row, col);
				if (tmp < sad12.at<double>(row, col))
				{
					sad12.at<double>(row, col) = tmp;
					depth12.at<double>(row, col) = d;
				}

				tmp = diff13.at<double>(row, col);
				if (tmp < sad13.at<double>(row, col))
				{
					sad13.at<double>(row, col) = tmp;
					depth13.at<double>(row, col) = d;
				}

				tmp = diff3.at<double>(row, col);
				if (tmp < sad3.at<double>(row, col))
				{
					sad3.at<double>(row, col) = tmp;
					depth3.at<double>(row, col) = d;
				}
			}
		}
		if (i % 50 == 0)
			cout << "depth: " << d << " out of " << saturated_depth
					<< " is finished." << endl;
	}

	double min, max;
	minMaxLoc(depth12, &min, &max);
	depth12.convertTo(depth12, CV_8UC1, 255.0 / (max - min),
			-min * 255.0 / (max - min));

	minMaxLoc(depth13, &min, &max);
	depth13.convertTo(depth13, CV_8UC1, 255.0 / (max - min),
			-min * 255.0 / (max - min));

	minMaxLoc(depth3, &min, &max);
	depth3.convertTo(depth3, CV_8UC1, 255.0 / (max - min),
			-min * 255.0 / (max - min));

	imshow("img1", depth12);
	imshow("img3", depth3);
	imshow("img2", depth13);
	cout
			<< "\nDepth map are shown. img1 for image1-2. img2 for image1-3. img3 for image1-2-3."
			<< endl;
	cout << "Press q to exit.\n" << endl;
	char key = waitKey(0);
	while (key != 'q')
	{
		key = waitKey(0);
	}
}
