CC = g++

CFLAGS = -std=c++1y -O3 -g3 
INCFLAGS = -I/usr/include -I/usr/local/include 
LDFLAGS = -L/usr/lib -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_calib3d -lopencv_features2d -lopencv_xfeatures2d -lopencv_flann -lopencv_imgproc

RM = /bin/rm -f
all:  multiview_stereo
multiview_stereo: main.o calibration.o depth_cal.o feature_select.o
	$(CC) -o multivew_stereo ./build/main.o ./build/calibration.o ./build/depth_cal.o ./build/feature_select.o $(INCFLAGS) $(LDFLAGS) 
main.o: ./src/main.cpp
	mkdir -p ./build
	$(CC) $(CFLAGS) $(INCFLAGS) -Wall -c -o ./build/main.o ./src/main.cpp
calibration.o:./src/calibration.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) -Wall -c -o ./build/calibration.o ./src/calibration.cpp
depth_cal.o:./src/depth_cal.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) -Wall -c -o ./build/depth_cal.o ./src/depth_cal.cpp
feature_select.o:./src/feature_select.cpp
	$(CC) $(CFLAGS) $(INCFLAGS) -Wall -c -o ./build/feature_select.o ./src/feature_select.cpp
directories: ${MKDIR_P} ./build
clean: 
	$(RM) multiview_stereo 
	$(RM) -r ./build/ 

