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

## Ubuntu 16.04 / 16.10 / 17.04

### 1) Dependencies

a) Add GPG key and repository to your 'sources.list.d'

	$ wget -O - https://repo.dfz.pt/apt/dfz_apt.key | sudo apt-key add -
	$ echo 'deb https://repo.dfz.pt/apt/ubuntu $(lsb_release -sc) main' | sudo tee /etc/apt/sources.list.d/naazgull.list

b) Install zapata JSON library

	$ sudo apt update
	$ sudo apt install zapata-base zapata-json

### 2) Install Godwin

	$ sudo apt install godwin
