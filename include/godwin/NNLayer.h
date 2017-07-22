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
#pragma once

#include <zapata/json.h>
#include <godwin/Matrix.h>

using namespace std;
#if !defined __APPLE__
using namespace __gnu_cxx;
#endif

namespace gdw {

	class NNLayer;

	typedef size_t index_t;
	typedef gdw::NNLayer nn;
	typedef arma::field< arma::mat > layer;
	typedef std::tuple< gdw::layer, gdw::layer > layer_delta;
	
	class neural_net : public std::shared_ptr< gdw::NNLayer > {
	public:
		neural_net();
		neural_net(zpt::json _sizes);
		neural_net(gdw::NNLayer* _target);
		virtual ~neural_net();
	};
	
	class NNLayer {
	public:
		NNLayer();
		NNLayer(zpt::json _sizes);
		virtual ~NNLayer();

		virtual void set_size(zpt::json _sizes);
			
		virtual zpt::json network();
		virtual gdw::layer& biases();
		virtual gdw::layer& b();
		virtual gdw::layer& weights();
		virtual gdw::layer& w();
		virtual double learning_rate();
		virtual void learning_rate(double _learning_rate);
		virtual void random_limits(double _min, double _max);
		
		virtual void set_feed_forward_lambda(zpt::lambda _function);
		virtual void set_feed_forward_lambda(gdw::index_t _layer, zpt::lambda _function);
		virtual void set_back_propagate_lambda(zpt::lambda _function);
		virtual void set_back_propagate_lambda(gdw::index_t _layer, zpt::lambda _function);

		virtual void push(gdw::index_t _layer, std::size_t _n_neurons);
		
		virtual void wire(zpt::json _network);
		virtual void wire(std::string _network_serialized);
		virtual void wire(std::istream _input_stream);

		virtual std::string snapshot();
		virtual void snapshot(std::ostream _output_stream);

		virtual zpt::json train(zpt::json _batch, std::size_t _training_batch_size, std::size_t _n_epochs);
		virtual zpt::json feed_forward(zpt::json _input);
	
	private:
		zpt::json __network;
		gdw::layer __w;
		gdw::layer __b;
		double __learning_rate;
		double __max_random;
		double __min_random;
		
		virtual gdw::layer_delta back_propagate(zpt::json _input, zpt::json _expected_output);
		virtual void builtins();

	};
	
	arma::mat sigmoid(arma::mat _z);
	arma::mat sigmoid_prime(arma::mat _z);
}
