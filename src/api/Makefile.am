lib_LTLIBRARIES = libgodwin-api.la

libgodwin_api_la_LIBADD = -lpthread -lzmq -lossp-uuid++ -lzapata-json -lzapata-http -lzapata-events -lzapata-addons -lzapata-zmq -lzapata-rest -lzapata-mongodb
libgodwin_api_la_LDFLAGS = -version-info 0:1:0 
libgodwin_api_la_CPPFLAGS = -std=c++11 -Wall -I../../include

libgodwin_api_la_SOURCES = \
godwin.cpp
