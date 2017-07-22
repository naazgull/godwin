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

# USAGE EXAMPLE

```c++
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>

#include <godwin/NNLayer.h>

using namespace std;
#if !defined __APPLE__
using namespace __gnu_cxx;
#endif

void learn_identity() {
	gdw::neural_net _nn({ 8, 3, 8 });
	zpt::json _hyper_params = {
		"learning_rate", 0.1,
		"epochs", 40000,
		"batch_size", 4,
		"random_limits", { 0, 0.1 }
	};	
	_nn->learning_rate((double) _hyper_params["learning_rate"]);
	_nn->random_limits((double) _hyper_params["random_limits"][0], (double) _hyper_params["random_limits"][1]);

	zpt::json _training_set = {
		{ { 1, 0, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0, 0 } },
		{ { 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0, 0 } },
		{ { 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0, 0 } },
		{ { 0, 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0, 0 } },
		{ { 0, 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 1, 0, 0, 0 } },
		{ { 0, 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0 } },
		{ { 0, 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0 } },
		{ { 0, 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 0, 0, 1 } }
	};
	_nn->train(_training_set, (size_t) _hyper_params["batch_size"], (size_t) _hyper_params["epochs"]);
	
	for (size_t _k = 0; _k != _training_set->arr()->size(); _k++) {
		zpt::json _denormalized = _nn->feed_forward(_training_set[_k][0]);
		double _max = 0;
		for (auto _n : _denormalized->arr()) {
			_max = (((double) _n) > _max ? (double) _n : _max);
		}
		_denormalized = _denormalized / zpt::json::floating(_max);
		zpt::json _result = zpt::json::array();
		for (auto _n : _denormalized->arr()) {
			_result << (int) _n;
		}
		_denormalized = _result;
		std::cout << std::string(_training_set[_k][0]) << " -> " << std::string(_denormalized) << endl <<  flush;
	}

	std::cout << "HIDDEN LAYER ->\n" << flush;
	for (size_t _col = 0; _col != _nn->w()(1).n_cols; _col++) {
		for (size_t _row = 0; _row != _nn->w()(1).n_rows; _row++){
			if (_nn->w()(1)(_row, _col) > 0) {
				std::cout << "1 " << flush;
			}
			else {
				std::cout << "0 " << flush;
			}
		}
		std::cout << endl << flush;
	}
}

int main(int argc, char* argv[]) {
	try {
		learn_identity();
	}
	catch (zpt::assertion& _e) {
		std::cout << _e.what() << endl << flush;
	}
	return 0;
}

```
