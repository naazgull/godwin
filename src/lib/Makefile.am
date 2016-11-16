lib_LTLIBRARIES = libgodwin.la

libgodwin_la_LIBADD = -lpthread -lzmq -lossp-uuid++ -lzapata-base -lzapata-json -larmadillo
libgodwin_la_LDFLAGS = -version-info 0:1:0 
libgodwin_la_CPPFLAGS = -std=c++11 -Wall -I../../include

libgodwin_la_SOURCES = \
Matrix.cpp \
NNLayer.cpp

bin_PROGRAMS = gdw-test

gdw_test_LDADD =  -lpthread -lzmq -lossp-uuid++ -lzapata-base -lzapata-json -larmadillo -lgodwin
gdw_test_LDFLAGS =  -L../lib/.libs
gdw_test_CPPFLAGS = -O3 -std=c++11 -I../../include

gdw_test_SOURCES = \
./test.cpp
