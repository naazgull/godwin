lib_LTLIBRARIES = libgodwin.la

libgodwin_la_LIBADD = -lpthread -lzmq -lossp-uuid++ -lzapata-json -lzapata-http -lzapata-events -lzapata-addons -lzapata-zmq -lzapata-rest -lzapata-mongodb
libgodwin_la_LDFLAGS = -version-info 0:1:0 
libgodwin_la_CPPFLAGS = -std=c++11 -Wall -I../../include

libgodwin_la_SOURCES = \
godwin.cpp
