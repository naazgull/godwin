/*
The MIT License (MIT)

Copyright (c) 2014 n@zgul <naazgull@dfz.pt>

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

int main(int argc, char* argv[]) {
	gdw::neural_net _nn;

	_nn->learning_rate(0.05);
	_nn->weight_generation_limits(0.001, 0.05);
	
	_nn->set_value_lambda(zpt::lambda("gdw::nn::linear", 1));
	_nn->set_threshold_lambda(zpt::lambda("gdw::nn::sigmoid", 1));
	_nn->set_backpropagation_lambda(zpt::lambda("gdw::nn::gradient_descent::sigma", 2));

	gdw::index_t _layer_0 = _nn->push({ "is_input", true }, 8);
	gdw::index_t _layer_1 = _nn->push(zpt::json::object(), 3);
	gdw::index_t _layer_2 = _nn->push({ "is_output", true }, 8);

	zpt::json _l0_nodes = zpt::json::array();
	for (size_t _i = _layer_0; _i != _layer_1; _i++) {
		_l0_nodes << _i;
	}
	
	zpt::json _l1_nodes = zpt::json::array();
	for (size_t _i = _layer_1; _i != _layer_2; _i++) {
		_nn->wire(_i, _l0_nodes);
		_l1_nodes << _i;
	}
	
	for (size_t _i = _layer_2; _i != _layer_2 + 8; _i++) {
		_nn->wire(_i, _l1_nodes);
	}

	std::cout << *_nn->matrix(WEIGHTS) << endl << flush;

	zpt::json _training_set(
		{
			{ zpt::array, 1, 0, 0, 0, 0, 0, 0, 0 },
			{ zpt::array, 0, 1, 0, 0, 0, 0, 0, 0 },
			{ zpt::array, 0, 0, 1, 0, 0, 0, 0, 0 },
			{ zpt::array, 0, 0, 0, 1, 0, 0, 0, 0 },
			{ zpt::array, 0, 0, 0, 0, 1, 0, 0, 0 },
			{ zpt::array, 0, 0, 0, 0, 0, 1, 0, 0 },
			{ zpt::array, 0, 0, 0, 0, 0, 0, 1, 0 },
			{ zpt::array, 0, 0, 0, 0, 0, 0, 0, 1 },
		}
	);
	std::random_device _rd;
	std::mt19937 _gen(_rd());
	std::uniform_int_distribution<int> _dist(0, 8);
	for (size_t _epoch = 0; _epoch != 80000; _epoch++) {
		int _index = _dist(_gen);
		_nn->train(_training_set[_index], _training_set[_index]);
		std::cout << "\r" << _epoch << flush;
	}
	std::cout << endl << flush;
	
	std::cout << *_nn->matrix(WEIGHTS) << endl << flush;
	std::cout << *_nn->matrix(SIGMAS) << endl << flush;

	for (auto _entry : _training_set->arr()) {
		std::cout << std::string(_entry) << " -> " << std::string(_nn->classify(_entry)) << endl << flush;
	}
	
	return 0;
}
