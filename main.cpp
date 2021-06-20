#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<conio.h>
#include<stdio.h>
#include<windows.h>
#include<queue>

using namespace cv;
using namespace std;

int barsThreshold = 240;
int verticalRatio = 8;
int horizontalRatio = 10;
int frameAnalyzeDelay = 5;
const int frameAnalyzeRows = 3;
const int frameAnalyzeColumns = 3;
float histSufficientMatch = 0.98;
float monocolorPixelThreshold = 0.90;
float saturationThreshold = 0.05;
float noiseThreshold = 0.5;

void updateVerticalRatio(int, void*) {
	verticalRatio = getTrackbarPos("VerticalRatio", "Result");
}
void updateHorizontalRatio(int, void*) {
	horizontalRatio = getTrackbarPos("HorizontalRatio", "Result");
}

bool isNoisy(Mat frame) {
	Mat color;
	int sumOfPixels = 0;
	cvtColor(frame, color, COLOR_BGR2HSV);
	for (int x = 0; x < color.rows; x++)
		for (int y = 0; y < color.cols; y++)
			if (color.at<Vec3b>(x, y)[1] < int(255 * saturationThreshold))
				sumOfPixels++;
	cout << sumOfPixels << endl;
	if (sumOfPixels / (color.rows * color.cols) > noiseThreshold)
		return true;
	else
		return false;
}

Mat detectBlackBars(Mat frame) {
	Mat leftVertical = frame(Rect(0, 0, frame.size[1] / verticalRatio, frame.size[0]));
	cvtColor(leftVertical, leftVertical, COLOR_RGB2GRAY);
	threshold(leftVertical, leftVertical, barsThreshold, 255, THRESH_BINARY);
	Mat rightVertical = frame(Rect(frame.size[1] - (frame.size[1] / verticalRatio) + 1, 0, frame.size[1] / verticalRatio - 1, frame.size[0]));
	cvtColor(rightVertical, rightVertical, COLOR_RGB2GRAY);
	threshold(rightVertical, rightVertical, barsThreshold, 255, THRESH_BINARY);
	Mat upperHorizontal = frame(Rect(0, 0, frame.size[1], frame.size[0] / horizontalRatio));
	cvtColor(upperHorizontal, upperHorizontal, COLOR_RGB2GRAY);
	threshold(upperHorizontal, upperHorizontal, barsThreshold, 255, THRESH_BINARY);
	Mat lowerHorizontal = frame(Rect(0, frame.size[0] - (frame.size[0] / horizontalRatio) + 1, frame.size[1], frame.size[0] / horizontalRatio - 1));
	cvtColor(lowerHorizontal, lowerHorizontal, COLOR_RGB2GRAY);
	threshold(lowerHorizontal, lowerHorizontal, barsThreshold, 255, THRESH_BINARY);
	
	Mat resultFrame = frame.clone();
	if (!(countNonZero(leftVertical) + countNonZero(rightVertical))) {
		putText(resultFrame, "Wykryto pionowe czarne pasy", Point(frame.size[1] / verticalRatio + 10, 15), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2);
		rectangle(resultFrame, Point(0, 0), Point(frame.size[1]/8, frame.size[0]-5), Scalar(0, 255, 0), 5);
		rectangle(resultFrame, Point(frame.size[1] - (frame.size[1] / verticalRatio) + 1, 0), Point(frame.size[1] - 5, frame.size[0] - 5), Scalar(0, 255, 0), 5);
	}
	if (!(countNonZero(upperHorizontal) + countNonZero(lowerHorizontal))) {
		putText(resultFrame, "Wykryto poziome czarne pasy", Point(frame.size[1] / verticalRatio + 10, 45), FONT_HERSHEY_PLAIN, 1.25, Scalar(255, 0, 0), 2);
		rectangle(resultFrame, Point(0, 0), Point(frame.size[1], frame.size[0] / horizontalRatio - 5), Scalar(255, 0, 0), 5);
		rectangle(resultFrame, Point(0, frame.size[0] - (frame.size[0] / horizontalRatio) + 1), Point(frame.size[1] - 5, frame.size[0] - 5), Scalar(255, 0, 0), 5);
	}
	return resultFrame;
}

Mat detectSimilarFragments(Mat frame, Mat previousFrame) {
	Mat hsvFrame, hsvPreviousFrame, resultFrame, histFrame, histPreviousFrame;
	int channels[] = { 0, 1 };
	int histSize[] = { 50, 60 };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	
	cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
	cvtColor(previousFrame, hsvPreviousFrame, COLOR_BGR2HSV);
	calcHist(&hsvFrame, 1, channels, Mat(), histFrame, 2, histSize, ranges, true, false);
	//normalize(histFrame, histFrame, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsvPreviousFrame, 1, channels, Mat(), histPreviousFrame, 2, histSize, ranges, true, false);
	//normalize(histPreviousFrame, histPreviousFrame, 0, 1, NORM_MINMAX, -1, Mat());

	//cout << compareHist(histFrame, histPreviousFrame, 0) << endl;
	if (compareHist(histFrame, histPreviousFrame, 0) > histSufficientMatch) {
		resultFrame = frame.clone();
		rectangle(resultFrame, Point(0, 0), Point(resultFrame.size[1] - 5, resultFrame.size[0] - 5), Scalar(100, 100, 0), 5);
		return resultFrame;
	}
	else
		return frame;
}

void blackBarsExample(VideoCapture cap) {
	Mat frame, resultFrame;
	namedWindow("Window", WINDOW_AUTOSIZE);
	namedWindow("Result", WINDOW_AUTOSIZE);

	double dWidth = cap.set(CAP_PROP_FRAME_WIDTH, 640);
	double dHeight = cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	createTrackbar("VerticalRatio", "Result", &verticalRatio, 20, updateVerticalRatio);
	createTrackbar("HorizontalRatio", "Result", &horizontalRatio, 20, updateHorizontalRatio);

	while (true) {
		if (!cap.read(frame)) {
			cout << "\n\nKoniec filmu\n\n";
			break;
		}
		resultFrame = detectBlackBars(frame);
		imshow("Window", frame);
		imshow("Result", resultFrame);
		if (waitKey(15) == 27) {
			cap.release();
			destroyAllWindows();
			return;
		}
	}
}

void histogramAnalysisExample(VideoCapture cap) {
	Mat frame, previousFrame, resultFrame;
	queue<Mat> previousFrameList;
	namedWindow("Window", WINDOW_AUTOSIZE);
	namedWindow("Result", WINDOW_AUTOSIZE);

	double dWidth = cap.set(CAP_PROP_FRAME_WIDTH, 640);
	double dHeight = cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	while (true) {
		if (!cap.read(frame)) {
			cout << "\n\nKoniec filmu\n\n";
			break;
		}
		if (frameAnalyzeDelay <= previousFrameList.size()) {
			int height = frame.size[0];
			int width = frame.size[1];
			Mat results[frameAnalyzeRows][frameAnalyzeColumns];
			
			for (int x = 0; x < frameAnalyzeRows; x++)
				for (int y = 0; y < frameAnalyzeColumns; y++) 
					results[x][y] = detectSimilarFragments(frame(Rect((width / frameAnalyzeColumns) * y, (height / frameAnalyzeRows) * x, 
						width / frameAnalyzeColumns, height / frameAnalyzeRows)), previousFrameList.front()(Rect((width / frameAnalyzeColumns) * y, 
							(height / frameAnalyzeRows) * x, width / frameAnalyzeColumns, height / frameAnalyzeRows)));
			for (int x = 0; x < frameAnalyzeRows; x++)
				for (int y = 1; y < frameAnalyzeColumns; y++) 
					hconcat(results[x][0], results[x][y], results[x][0]);
			for (int x = 1; x < frameAnalyzeRows; x++)
				vconcat(results[0][0], results[x][0], results[0][0]);
			previousFrameList.pop();
			imshow("Result", results[0][0]);
		}
		imshow("Window", frame);
		if (waitKey(15) == 27) {
			cap.release();
			destroyAllWindows();
			return;
		}
		previousFrameList.push(frame.clone());
	}
}

void videoDistortionsExample(VideoCapture cap) {
	Mat frame, resultFrame;
	namedWindow("Window", WINDOW_AUTOSIZE);

	double dWidth = cap.set(CAP_PROP_FRAME_WIDTH, 640);
	double dHeight = cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	while (true) {
		if (!cap.read(frame)) {
			cout << "\n\nKoniec filmu\n\n";
			break;
		}
		resultFrame = frame.clone();
		int samePixelCounter = 0;
		Vec3b basePixel = frame.at<Vec3b>(frame.cols / 2, frame.rows / 2);
		for (int x = 0; x < frame.rows; x++)
			for (int y = 0; y < frame.cols; y++)
				if (basePixel == frame.at<Vec3b>(x, y)) 
					samePixelCounter++;
		if (samePixelCounter / (frame.rows * frame.cols) > monocolorPixelThreshold)
			putText(resultFrame, "Wykryto pusta klatke", Point(10, 30), FONT_HERSHEY_PLAIN, 1.25, Scalar(150, 0, 150), 2);
		if (isNoisy(frame)) 
			putText(resultFrame, "Wykryto zakłócenia", Point(10, 45), FONT_HERSHEY_PLAIN, 1.25, Scalar(150, 0, 150), 2);
		imshow("Window", resultFrame);
		if (waitKey(15) == 27) {
			cap.release();
			destroyAllWindows();
			return;
		}
	}
}

void cameraExample(VideoCapture cap) {
	Mat frame, resultFrame, distortionFrame;
	queue<Mat> previousFrameList;
	namedWindow("Window", WINDOW_AUTOSIZE);
	namedWindow("Result", WINDOW_AUTOSIZE);
	namedWindow("Histogram", WINDOW_AUTOSIZE);
	namedWindow("Distortions", WINDOW_AUTOSIZE);

	double dWidth = cap.set(CAP_PROP_FRAME_WIDTH, 640);
	double dHeight = cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	while (true) {
		cap >> frame;
		distortionFrame = frame.clone();
		resultFrame = detectBlackBars(frame);

		if (frameAnalyzeDelay <= previousFrameList.size()) {
			int height = frame.size[0];
			int width = frame.size[1];
			Mat images[frameAnalyzeRows][frameAnalyzeColumns];
			Mat previousImages[frameAnalyzeRows][frameAnalyzeColumns];
			Mat results[frameAnalyzeRows][frameAnalyzeColumns];

			for (int x = 0; x < frameAnalyzeRows; x++)
				for (int y = 0; y < frameAnalyzeColumns; y++)
					results[x][y] = detectSimilarFragments(frame(Rect((width / frameAnalyzeColumns) * y, (height / frameAnalyzeRows) * x,
						width / frameAnalyzeColumns, height / frameAnalyzeRows)), previousFrameList.front()(Rect((width / frameAnalyzeColumns) * y,
							(height / frameAnalyzeRows) * x, width / frameAnalyzeColumns, height / frameAnalyzeRows)));
			for (int x = 0; x < frameAnalyzeRows; x++)
				for (int y = 1; y < frameAnalyzeColumns; y++)
					hconcat(results[x][0], results[x][y], results[x][0]);
			for (int x = 1; x < frameAnalyzeRows; x++)
				vconcat(results[0][0], results[x][0], results[0][0]);
			previousFrameList.pop();
			imshow("Histogram", results[0][0]);
		}

		int samePixelCounter = 0;
		Vec3b basePixel = frame.at<Vec3b>(frame.cols / 2, frame.rows / 2);
		for (int x = 0; x < frame.rows; x++)
			for (int y = 0; y < frame.cols; y++)
				if (basePixel == frame.at<Vec3b>(x, y))
					samePixelCounter++;
		if (samePixelCounter / (frame.rows * frame.cols) > monocolorPixelThreshold)
			putText(distortionFrame, "Wykryto pusta klatke", Point(10, 30), FONT_HERSHEY_PLAIN, 1.25, Scalar(150, 0, 150), 2);
		if (isNoisy(frame))
			putText(distortionFrame, "Wykryto zakłócenia", Point(10, 45), FONT_HERSHEY_PLAIN, 1.25, Scalar(150, 0, 150), 2);

		imshow("Distortions", distortionFrame);
		imshow("Result", resultFrame);
		imshow("Window", frame);
		if (waitKey(15) == 27) {
			cap.release();
			destroyAllWindows();
			return;
		}
		previousFrameList.push(frame.clone());
	}
}

int main() {
	VideoCapture cap;
	int choice;
	cout << "Podaj wybor funkcji programu:" << endl;
	cout << "1 - wykrywanie czarnych marginesow na przykladowym pliku wideo" << endl;
	cout << "2 - wykrywanie stalych fragmentow w klatkach na przykladowym pliku wideo" << endl;
	cout << "3 - wykrywanie pustych i znieksztalconych klatek na przykladowym pliku wideo" << endl;
	cout << "4 - test wszystkich funkcji na obrazie z kamerki" << endl;
	cout << "5 - wyjscie" << endl;
	cin >> choice;
	switch (choice) {
	case 1:
		cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/blackBarsSample.mp4");
		//cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/blackBarsSample2.mp4");
		if (!cap.isOpened()) {
			cout << "\n\nNie mozna otworzyc pliku\n\n";
			return 0;
		}
		blackBarsExample(cap);
		break;
	case 2:
		cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/blackBarsSample2.mp4");
		//cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/bike.wm");
		if (!cap.isOpened()) {
			cout << "\n\nNie mozna otworzyc pliku\n\n";
			return 0;
		}
		histogramAnalysisExample(cap);
		break;
	case 3:
		cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/blackScreenSample.mp4");
		//cap.open("C:/Users/Cichy/Desktop/Studia/widzenie/noiseSample.mp4");
		if (!cap.isOpened()) {
			cout << "\n\nNie mozna otworzyc pliku\n\n";
			return 0;
		}
		videoDistortionsExample(cap);
		break;
	case 4:
		cap.open(2);
		if (!cap.isOpened())
			cap.open(0);
		cameraExample(cap);
	default:
		return 0;
	}
	return 0;
}
