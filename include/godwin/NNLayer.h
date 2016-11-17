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
#pragma once

#include <zapata/json.h>
#include <godwin/Matrix.h>

using namespace std;
#if !defined __APPLE__
using namespace __gnu_cxx;
#endif

#define N_MATRIX 4
#define WEIGHTS 0
#define OUTPUTS 1
#define DELTAS 2
#define SIGMAS 3

namespace gdw {

	class NNLayer;

	typedef size_t index_t;
	typedef gdw::NNLayer nn;
	
	class neural_net : public std::shared_ptr< gdw::NNLayer > {
	public:
		neural_net();
		neural_net(gdw::NNLayer* _target);
		virtual ~neural_net();
	};
	
	class NNLayer {
	public:
		NNLayer();
		virtual ~NNLayer();

		virtual zpt::json network();
		virtual gdw::mat_ptr matrix(gdw::index_t _which);
		virtual double learning_rate();
		virtual void learning_rate(double _learning_rate);
		
		virtual void set_value_lambda(zpt::lambda _function);
		virtual void set_value_lambda(gdw::index_t _neuron, zpt::lambda _function);
		virtual void set_threshold_lambda(zpt::lambda _function);
		virtual void set_threshold_lambda(gdw::index_t _neuron, zpt::lambda _function);
		virtual void set_backpropagation_lambda(zpt::lambda _function);
		virtual void set_backpropagation_lambda(gdw::index_t _neuron, zpt::lambda _function);

		virtual gdw::index_t push(zpt::json _neuron, size_t _n_neurons = 1);
		
		virtual void wire(gdw::index_t _neuron, zpt::json _inbound);
		virtual void wire(zpt::json _network);
		virtual void wire(std::string _network_serialized);
		virtual void wire(std::istream _input_stream);

		virtual std::string snapshot();
		virtual void snapshot(std::ostream _output_stream);

		virtual void train(zpt::json _input, zpt::json _expected_output);
		virtual zpt::json classify(zpt::json _input);

		static std::string __matrix_names[N_MATRIX];

	private:
		zpt::json __network;
		std::vector< gdw::mat_ptr > __matrix;
		double __learning_rate;
		
		virtual zpt::json classify(arma::uvec& _to_process);
		virtual void adjust(zpt::json _input, zpt::json _expected_output);
		virtual void adjust(arma::uvec& _to_process);

	};
	
}
