#include <iostream>
#include "cv.hpp"

using namespace std;
using namespace cv;

void move(Mat &input, Mat &output, int x_move, int y_move) {
	for (int y = y_move; y < output.rows; y++)
		for (int x = x_move; x < output.cols; x++) {
			Vec3b temp = input.at<Vec3b>(y - y_move, x - x_move);
			output.at<Vec3b>(y, x) = temp;
		}
}

void imageOnImage(Mat &small_image, Mat &big_image) {
	for (int y = 0; y < small_image.rows; y++) {
		for (int x = 0; x < small_image.cols; x++) {
			big_image.at<Vec3b>(y, x) = small_image.at<Vec3b>(y, x);
		}
	}
}


struct basic
{
	vector<Mat> doorlist;
	int number = 17;
	vector<KeyPoint> keypoint1[17] = {};
	vector<KeyPoint> prevKeypoint;
	Mat descriptor1[17] = {};
	Mat prevDescriptor, currDescriptor;
	vector<Mat> facelist;
};


void pushquery(string door, string face, basic& basic)
{
	for (int i = 1; i < basic.number + 1; i++) {
		Ptr<ORB> orbF = ORB::create(1000);
		door[10] = face[10] = (char)(i + 64);
		Mat image = imread(door);
		resize(image, image, Size(350, 700));
		basic.doorlist.push_back(image);

		Mat image2 = imread(face);
		resize(image2, image2, image.size() / 3);
		basic.facelist.push_back(image2);

		orbF->detectAndCompute(basic.doorlist[i - 1], noArray(), basic.keypoint1[i - 1], basic.descriptor1[i - 1]);

	}
}

Mat preprocessingvideo(Mat frame) {
	transpose(frame, frame); // rotate 90'
	flip(frame, frame, 1); // flip horizontally
	resize(frame, frame, Size(350, 700));
	return frame;
}





int main() {
	vector<vector<DMatch>>matches;
	vector<DMatch> goodMatches;
	VideoCapture cap("prof_video_4.mp4");
	Mat frame, transform, faceimg, background, result;
	int cnt = 0;
	int curr = 0;

	string door = "prof_door_n.jpg";
	string face = "prof_face_n.png";
	basic basic;

	int fps = cap.get(CAP_PROP_FPS);

	cap >> frame;
	int width = frame.cols;
	int height = frame.rows;

	pushquery(door, face, basic);

	Mat imgMatches, H, HH, match;
	Mat query_mask2;
	Mat frame_mask = Mat::zeros(frame.size(), frame.type());
	vector<Point2f> inputArray(4), outputArray(4);
	//imshow("face", basic.facelist[0]);

	while (1) {
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		frame = preprocessingvideo(frame);

		Mat image, descriptors2;
		Mat mask = Mat(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
		vector<KeyPoint> keypoint2;
		Ptr<ORB> orbF = ORB::create(1000);
		BFMatcher matcher(NORM_HAMMING);

		vector<Point2f> obj, scene;

		int i, k;
		float nndrRatio;
		image = frame.clone();

		if (curr < 5) {
			orbF->detectAndCompute(image, noArray(), keypoint2, descriptors2);
			basic.currDescriptor = descriptors2.clone();

			k = 2;
			if (descriptors2.rows >= k) {
				matcher.knnMatch(basic.descriptor1[cnt % basic.number], descriptors2, matches, k);

				nndrRatio = 0.6f;
				for (i = 0; i < matches.size(); i++)
				{
					if (matches.at(i).size() == 2
						&& matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
					{
						goodMatches.push_back(matches[i][0]);
					}
				}

				drawMatches(basic.doorlist[cnt % basic.number], basic.keypoint1[cnt % basic.number], image, keypoint2, goodMatches, imgMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				if (goodMatches.size() > 10)
				{
					curr++;
					for (i = 0; i < goodMatches.size(); i++)
					{
						obj.push_back(basic.keypoint1[cnt % basic.number][goodMatches[i].queryIdx].pt);
						scene.push_back(keypoint2[goodMatches[i].trainIdx].pt);
					}
					H = findHomography(obj, scene, RANSAC);

					if (H.rows != 0 && H.dims != 0) {
						//query warping
						resize(basic.facelist[cnt % basic.number], basic.facelist[cnt % basic.number], Size(basic.doorlist[cnt % basic.number].cols / 3, basic.doorlist[cnt % basic.number].rows / 3));
						Mat query_mask1 = Mat::zeros(basic.doorlist[cnt % basic.number].size(), basic.doorlist[cnt % basic.number].type());
						query_mask2 = query_mask1.clone();
						imageOnImage(basic.facelist[cnt % basic.number], query_mask1);
						move(query_mask1, query_mask2, basic.doorlist[cnt % basic.number].cols / 3, basic.doorlist[cnt % basic.number].rows / 3);

						inputArray[0] = Point2f(query_mask2.cols / 3, query_mask2.rows / 3);
						inputArray[1] = Point2f((query_mask2.cols / 3) * 2, query_mask2.rows / 3);
						inputArray[2] = Point2f((query_mask2.cols / 3) * 2, (query_mask2.rows / 3) * 2);
						inputArray[3] = Point2f(query_mask2.cols / 3, (query_mask2.rows / 3) * 2);

						perspectiveTransform(inputArray, outputArray, H);

						warpPerspective(query_mask2, frame_mask, H, frame.size());
						frame_mask.copyTo(frame, frame_mask);
						cnt--;
					}
				}
				cnt++;
			}

			imshow("imgMatches", imgMatches);

			matches.clear();
			goodMatches.clear();
			basic.prevDescriptor = descriptors2.clone();
			basic.prevKeypoint = keypoint2;
		}

		else {
			orbF->detectAndCompute(image, noArray(), keypoint2, descriptors2);
			basic.currDescriptor = descriptors2.clone();
			vector<Point2f> outputArray2(4);

			k = 2;
			if (descriptors2.rows >= k) {
				matcher.knnMatch(basic.prevDescriptor, basic.currDescriptor, matches, k);

				nndrRatio = 0.6f;
				for (i = 0; i < matches.size(); i++)
				{
					if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
					{
						goodMatches.push_back(matches[i][0]);
					}
				}

				if (goodMatches.size() > 15)
				{
					curr++;
					for (i = 0; i < goodMatches.size(); i++)
					{
						obj.push_back(basic.prevKeypoint[goodMatches[i].queryIdx].pt);
						scene.push_back(keypoint2[goodMatches[i].trainIdx].pt);
					}
					HH = findHomography(obj, scene, RANSAC);

					if (HH.rows != 0 && HH.dims != 0) {
						perspectiveTransform(outputArray, outputArray2, HH);

						Mat transform = getPerspectiveTransform(inputArray, outputArray2);

						warpPerspective(query_mask2, frame_mask, transform, frame.size());
						frame_mask.copyTo(frame, frame_mask);
					}
				}

				else
					curr = 0;
			}

			matches.clear();
			goodMatches.clear();
			basic.prevDescriptor = descriptors2.clone();
			basic.prevKeypoint = keypoint2;
			outputArray = outputArray2;
			outputArray2.clear();
		}


		//resize(frame, frame, Size(width, height));
		imshow("frame", frame);
		waitKey(1000 / fps);
	}

	return 0;
}