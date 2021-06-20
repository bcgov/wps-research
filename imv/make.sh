g++ 	-c -w  -O4 -fno-sched-spec 	imv.cpp -g &
g++ 	-c -w  -O4 -fno-sched-spec 	newzpr.cpp -g &
g++ 	-c -w  -O4 -fno-sched-spec 	my_math.cpp -g &
g++ 	-c -w  -O4 -fno-sched-spec 	image.cpp -g &
g++ 	-c -w  -O4 -fno-sched-spec 	util.cpp -g &
wait
g++ 	imv.o newzpr.o my_math.o	image.o	util.o	-o	imv.exe  -lm -lpthread  -lGL -lGLU -lglut -g
