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
	//gdw::neural_net _nn;

	zpt::json _test({ "is_input", true, "f", zpt::json::lambda("gdw::nn::linear", 1) });
	std::cout << zpt::pretty(_test) << endl << flush;
	std::cout << zpt::pretty(_test) << endl << flush;
	std::cout << zpt::pretty(_test) << endl << flush;
	
	//_nn->push({ "is_intput", true }, 3);
	//_nn->push(zpt::json::object());
	//_nn->push({ "is_output", true }, 3);

	//_nn->set_value_lambda(zpt::lambda("gdw::nn::linear", 1));
	//_nn->set_threshold_lambda(zpt::lambda("gdw::nn::sigmoid", 1));
	//_nn->set_backpropagation_lambda(zpt::lambda("gdw::nn::stochastic_gradient_descent", 4));
	//std::cout << zpt::pretty(_nn->network()) << endl << flush;
	//std::cout << zpt::pretty(_nn->network()) << endl << flush;
	
	return 0;
}
