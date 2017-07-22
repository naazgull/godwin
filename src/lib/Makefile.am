lib_LTLIBRARIES = libgodwin.la

libgodwin_la_LIBADD = -lpthread -lossp-uuid++ -lzapata-base -lzapata-json -larmadillo
libgodwin_la_LDFLAGS = -version-info 9:3:9 
libgodwin_la_CPPFLAGS = -std=c++14 -Wall -I../../include

libgodwin_la_SOURCES = \
Matrix.cpp \
NNLayer.cpp \
AGLayer.cpp

bin_PROGRAMS = gdw-test

gdw_test_LDADD =  -lpthread -lossp-uuid++ -lzapata-base -lzapata-json -larmadillo -lgodwin
gdw_test_LDFLAGS =  -L../lib/.libs
gdw_test_CPPFLAGS = -O3 -std=c++14 -I../../include

gdw_test_SOURCES = \
./test.cpp
