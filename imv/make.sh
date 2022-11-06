g++ 	-c -w  -O4  	imv.cpp  &
g++ 	-c -w  -O4  	newzpr.cpp  &
g++ 	-c -w  -O4  	my_math.cpp  &
g++ 	-c -w  -O4  	image.cpp  &
g++ 	-c -w  -O4  	util.cpp  &
wait
g++ -O4 imv.o newzpr.o my_math.o	image.o	util.o	-o	imv.exe  -lm -lpthread  -lGL -lGLU -lglut 
