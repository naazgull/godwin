Godwin
================================

_Godwin_ is a C++14/C++17 styled library for training **Artificial Neural Networks**,
built upon the [Armadillo](http://arma.sourceforge.net) matrix library and the 
[Zapata](https://github.com/naazgull/zapata) REST and JSON support library.

The main feaures are:

- Store and load the **_ANN_** using JSON
- Use _lambda_ functions to define handlers for feed-forward, back-propagate and training
stages

# INSTALLATION

## Ubuntu 16.04 / 16.10

### 1) Dependencies

a) Install libzmq, libczmq and sodium from Ubuntu repositories:

	$ sudo apt-get install libzmq5 libczmq3 libsodium18 libarmadillo-dev

b) Install [libcurve](https://github.com/zeromq/libcurve) from Github:

	$ git clone git://github.com/zeromq/libcurve.git
	$ cd libcurve
	$ sh autogen.sh
	$ ./autogen.sh
	$ ./configure && make check
	$ sudo make install
	$ sudo ldconfig
	$ cd ..

c) Add GPG key and repository to your 'sources.list.d'

	$ wget -O - https://repo.dfz.pt/apt/dfz_apt.key | sudo apt-key add -
	$ echo 'deb https://repo.dfz.pt/apt/ubuntu xenial main' | sudo tee /etc/apt/sources.list.d/naazgull.list

d) Install zapata

	$ sudo apt-get update
	$ sudo apt-get install zapata-base zapata-json zapata-http zapata-addons zapata-events zapata-zmq zapata-rest

### 2) Install Godwin

	$ sudo apt-get update
	$ sudo apt-get install godwin
