OBJS = newzpr.o dem.o misc.o
CC = g++ -std=c++11
DEBUG = # -g  # uncomment the g for DEBUG mode
CFLAGS =  -c -w $(DEBUG) -O3 misc.h misc.cpp -march=native -ffast-math -funroll-loops -DNDEBUG
LFLAGS =  $(DEBUG) -lm -lGLEW -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LFLAGS += -lGL -lGLU -lglut
endif
ifeq ($(UNAME_S),Darwin)
    LFLAGS += -framework OpenGL -framework glut
endif

zpr: $(OBJS)
    $(CC)   $(OBJS) -o  dem  $(LFLAGS)
newzpr.o: newzpr.cpp  newzpr.h
    $(CC) $(CFLAGS) newzpr.cpp
dem.o:  dem.cpp newzpr.h  newzpr.cpp
    $(CC)       $(CFLAGS)   dem.cpp
