# to debug:
# turn optimization flag to -O1 or -O2
# use debug flag -ggdb
# gdb
# file imv.exe
# run

# m3ta3: reimagination of a (late 2011, early 2012) personal, primordial visualization library that inspired further developments at UVic, CFS and elsewhere
OBJS = imv.o newzpr.o my_math.o	image.o	util.o
CC = g++ # -g # i686-apple-darwin11-llvm-g++-4.2 #-mp-4.7
DEBUG = # -ggdb -O1
CFLAGS =   -c -w $(DEBUG) -O3 #-O0 # -fno-sched-spec # -bind_at_load `pkg-config --cflags opencv`     
#LFLAGS =  $(DEBUG) -lm -lpthread  -framework Cocoa -framework GLUT -framework OpenGL # -lpython2.5 -lm -L/usr/lib/python2.5/config `pkg-config --libs opencv`
LFLAGS =  $(DEBUG) -lm -lpthread  -lGL -lGLU -lglut #-framework Cocoa -framework GLUT -framework OpenGL # -lpython2.5 -lm -L/usr/lib/python2.5/config `pkg-config --libs opencv`

zpr:	$(OBJS)
	$(CC)	$(OBJS)	-o	imv.exe $(LFLAGS)

my_math.o:	my_math.cpp	my_math.h
	$(CC)	$(CFLAGS)	my_math.cpp

newzpr.o:	newzpr.cpp	newzpr.h
	$(CC)	$(CFLAGS)	newzpr.cpp

image.o:	image.cpp	image.h
	$(CC)	$(CFLAGS)	image.cpp

util.o:		util.cpp
	$(CC)	$(CFLAGS)	util.cpp

imv.o:		imv.cpp
	$(CC)	$(CFLAGS)	imv.cpp
