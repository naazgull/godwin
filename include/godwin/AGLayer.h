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

#define N_MATRIX 5
#define WEIGHTS 0
#define OUTPUTS 1
#define DELTAS 2
#define SIGMAS 3
#define UNITS 4

namespace gdw {

	class AGLayer;

	typedef size_t index_t;
	typedef gdw::AGLayer ag;
	
	class acyclic_graph : public std::shared_ptr< gdw::AGLayer > {
	public:
		acyclic_graph();
		acyclic_graph(zpt::json _sizes);
		acyclic_graph(gdw::AGLayer* _target);
		virtual ~acyclic_graph();
	};
	
	class AGLayer {
	public:
		AGLayer();
		AGLayer(zpt::json _sizes);
		virtual ~AGLayer();

		virtual zpt::json graph();
		virtual gdw::mat_ptr matrix(gdw::index_t _which);
		virtual double learning_rate();
		virtual void learning_rate(double _learning_rate);
		virtual double* weight_generation_limits();
		virtual void weight_generation_limits(double _lower, double _higher);
		
		virtual void set_value_lambda(zpt::lambda _function);
		virtual void set_value_lambda(gdw::index_t _neuron, zpt::lambda _function);
		virtual void set_threshold_lambda(zpt::lambda _function);
		virtual void set_threshold_lambda(gdw::index_t _neuron, zpt::lambda _function);	
		virtual void set_error_delta_lambda(zpt::lambda _function);
		virtual void set_error_delta_lambda(gdw::index_t _neuron, zpt::lambda _function);
		virtual void set_weight_adjust_lambda(zpt::lambda _function);
		virtual void set_weight_adjust_lambda(gdw::index_t _neuron, zpt::lambda _function);

		virtual gdw::index_t push(zpt::json _neuron, size_t _n_neurons = 1);
		
		virtual void wire(gdw::index_t _neuron, zpt::json _inbound);
		virtual void wire(zpt::json _network);
		virtual void wire(std::string _network_serialized);
		virtual void wire(std::istream _input_stream);

		virtual std::string snapshot();
		virtual void snapshot(std::ostream _output_stream);

		virtual zpt::json train(zpt::json _input, zpt::json _expected_output);
		virtual zpt::json classify(zpt::json _input);

		static std::string __matrix_names[N_MATRIX];

	private:
		zpt::json __graph;
		std::vector< gdw::mat_ptr > __matrix;
		double __learning_rate;
		double __weight_random_limits[2];
		
		virtual zpt::json classify(arma::uvec& _to_process);
		virtual void calculate_deltas(zpt::json _input, zpt::json _expected_output);
		virtual void calculate_deltas(arma::uvec& _to_process);
		virtual void adjust_weights(zpt::json _input);
		virtual void adjust_weights(arma::uvec& _to_process);
		virtual void builtins();

	};
	
}
