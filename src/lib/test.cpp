/*
The MIT License (MIT)

Copyright (c) 2014 n@zgul <n@zgul.me>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
	catch (zpt::AssertionException& _e) {
		std::cout << _e.what() << endl << flush;
	}
	return 0;
}
