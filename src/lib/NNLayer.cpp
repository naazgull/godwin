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
#include <godwin/NNLayer.h>
#include <algorithm>
#include <random>

gdw::neural_net::neural_net() : std::shared_ptr< gdw::NNLayer >(new gdw::NNLayer()) {
}

gdw::neural_net::neural_net(zpt::json _sizes) : std::shared_ptr< gdw::NNLayer >(new gdw::NNLayer(_sizes)) {
}

gdw::neural_net::neural_net(gdw::nn* _target) : std::shared_ptr< gdw::NNLayer >(_target) {
}

gdw::neural_net::~neural_net() {
}

gdw::NNLayer::NNLayer() : __learning_rate(0.05), __max_random(0.05), __min_random(-0.05) {
	this->builtins();
	this->__network = {
		"lambdas", {
			"defaults", {
				"train", zpt::lambda("gdw::nn::train::gradient_descent", 3),
				"feed_forward", zpt::lambda("gdw::nn::feed_forward::sigmoid", 1),
				"back_propagate", zpt::lambda("gdw::nn::back_propagate::gradient_descent", 2)
			}
		}
	};
}

gdw::NNLayer::NNLayer(zpt::json _sizes) : gdw::NNLayer() {
	this->set_size(_sizes);
}

gdw::NNLayer::~NNLayer() {
}

void gdw::NNLayer::set_size(zpt::json _sizes) {
	assertz(_sizes->type() == zpt::JSArray, "'_sizes' parameter is not a valid array", 412, 0);
	size_t _l = 0;

	this->__b.set_size(1, _sizes->arr()->size());
	this->__w.set_size(1, _sizes->arr()->size());
	
	for (auto _len : _sizes->arr()) {
		assertz(_len->type() == zpt::JSInteger, "'_sizes' parameter is not a valid array", 412, 0);
		this->push(_l, (size_t) _len);
		_l++;
	}
}

zpt::json gdw::NNLayer::network() {
	return this->__network;
}

double gdw::NNLayer::learning_rate() {
	return this->__learning_rate;
}

void gdw::NNLayer::learning_rate(double _learning_rate) {
	this->__learning_rate = _learning_rate;
}

void gdw::NNLayer::random_limits(double _min, double _max) {
	this->__max_random = _max;
	this->__min_random = _min;
	if (this->__b.n_cols != 0) {
		this->__b.for_each(
			[ & ] (arma::mat& _bias) -> void {
				_bias.randn();
				double _max = _bias.max();
				double _min = _bias.min();
				_bias = (_bias - _min) / (_max - _min) * (this->__max_random - this->__min_random) + this->__min_random;
			}
		);
	}
	if (this->__w.n_cols != 0) {
		this->__w.for_each(
			[ & ] (arma::mat& _weights) -> void {
				if (_weights.n_rows == 0) {
					return;
				}
				
				_weights.randn();
				double _max = _weights.max();
				double _min = _weights.min();
				_weights = (_weights - _min) / (_max - _min) * (this->__max_random - this->__min_random) + this->__min_random;
			}
		);
	}
}

gdw::layer& gdw::NNLayer::biases() {
	return this->__b;
}

gdw::layer& gdw::NNLayer::b() {
	return this->__b;
}

gdw::layer& gdw::NNLayer::weights() {
	return this->__w;
}

gdw::layer& gdw::NNLayer::w() {
	return this->__w;
}

void gdw::NNLayer::set_feed_forward_lambda(zpt::lambda _function) {
	this->__network["lambdas"]["defaults"] << "feed_forward" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_feed_forward_lambda(gdw::index_t _layer, zpt::lambda _function) {
	std::string _key = std::to_string(_layer);
	if (!this->__network["lambdas"]["layers"]->ok()) {
		this->__network["lambdas"] << "layers" << zpt::json::object();
	}
	if (!this->__network["lambdas"]["layers"][_key]->ok()) {
		this->__network["lambdas"]["layers"] << _key << zpt::json::object();
	}
	this->__network["lambdas"]["layers"][_key] << "feed_forward" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_back_propagate_lambda(zpt::lambda _function) {
	this->__network["lambdas"]["defaults"] << "back_propagate" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_back_propagate_lambda(gdw::index_t _layer, zpt::lambda _function) {
	std::string _key = std::to_string(_layer);
	if (!this->__network["lambdas"]["layers"]->ok()) {
		this->__network["lambdas"] << "layers" << zpt::json::object();
	}
	if (!this->__network["lambdas"]["layers"][_key]->ok()) {
		this->__network["lambdas"]["layers"] << _key << zpt::json::object();
	}
	this->__network["lambdas"]["layers"][_key] << "back_propagate" << zpt::json::lambda(_function);
}

void gdw::NNLayer::push(gdw::index_t _layer, std::size_t _n_neurons) {
	if (!this->__network->ok()) {
		this->__network = zpt::json::object();
	}

	arma::arma_rng::set_seed_random();
	
	this->__b(0, _layer) = arma::mat(_n_neurons, 1);
	this->__b(0, _layer).randn();
	{
		double _max = this->__b(0, _layer).max();
		double _min = this->__b(0, _layer).min();
		this->__b(0, _layer) = (this->__b(0, _layer) - _min) / (_max - _min) * (this->__max_random - this->__min_random) + this->__min_random;
	}

	if (_layer != 0) {
		this->__w(0, _layer) = arma::mat(_n_neurons, this->__b(0, _layer - 1).n_rows);
		this->__w(0, _layer).randn();
		double _max = this->__w(0, _layer).max();
		double _min = this->__w(0, _layer).min();
		this->__w(0, _layer) = (this->__w(0, _layer) - _min) / (_max - _min) * (this->__max_random - this->__min_random) + this->__min_random;
	}
}

void gdw::NNLayer::wire(zpt::json _network) {
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__network = _network;
	
	if (this->__network["biases"]->type() == zpt::JSString) {
		std::string _field_str(this->__network["biases"]->str());
		zpt::base64::decode(_field_str);
		std::istringstream _iss(_field_str);
		this->__b.load(_iss, arma::arma_binary);
	}
	if (this->__network["weights"]->type() == zpt::JSString) {
		std::string _field_str(this->__network["weights"]->str());
		zpt::base64::decode(_field_str);
		std::istringstream _iss(_field_str);
		this->__w.load(_iss, arma::arma_binary);
	}

}

void gdw::NNLayer::wire(std::string _network_serialized) {
	zpt::json _network(_network_serialized);
	this->wire(_network);
}

void gdw::NNLayer::wire(std::istream _input_stream) {
	zpt::json _network;
	_input_stream >> _network;
	this->wire(_network);
}

std::string gdw::NNLayer::snapshot() {
	{
		std::ostringstream _oss;
		this->__b.save(_oss, arma::arma_binary);
		std::string _field_str(_oss.str());
		zpt::base64::encode(_field_str);
		this->__network << "biases" << _field_str;
	}
	{
		std::ostringstream _oss;
		this->__w.save(_oss, arma::arma_binary);
		std::string _field_str(_oss.str());
		zpt::base64::encode(_field_str);
		this->__network << "weights" << _field_str;
	}
	return std::string(this->__network);
}

void gdw::NNLayer::snapshot(std::ostream _output_stream) {
	_output_stream << this->snapshot() << flush;
}

zpt::json gdw::NNLayer::feed_forward(zpt::json _input) {
	zpt::json _args({ _input });
	zpt::lambda _fn = (this->__network["lambdas"]["layers"]["0"]["feed_forward"]->type() == zpt::JSLambda ? this->__network["lambdas"]["layers"]["0"]["feed_forward"]->lbd() : this->__network["lambdas"]["defaults"]["feed_forward"]->lbd());
	return _fn(_args, zpt::context(this));
}

zpt::json gdw::NNLayer::train(zpt::json _batch, std::size_t _training_batch_size, std::size_t _n_epochs) {
	zpt::json _args({ _batch, zpt::json::ulong(_training_batch_size), zpt::json::ulong(_n_epochs) });
	zpt::lambda _fn = (this->__network["lambdas"]["layers"]["0"]["train"]->type() == zpt::JSLambda ? this->__network["lambdas"]["layers"]["0"]["train"]->lbd() : this->__network["lambdas"]["defaults"]["train"]->lbd());
	return _fn(_args, zpt::context(this));
}

gdw::layer_delta gdw::NNLayer::back_propagate(zpt::json _input, zpt::json _expected_output) {
	zpt::json _args({ _input, _expected_output });
	zpt::lambda _fn = (this->__network["lambdas"]["layers"]["0"]["back_propagate"]->type() == zpt::JSLambda ? this->__network["lambdas"]["layers"]["0"]["back_propagate"]->lbd() : this->__network["lambdas"]["defaults"]["back_propagate"]->lbd());

	gdw::layer _delta_nabla_b;
	_delta_nabla_b.copy_size(this->b());
	gdw::layer _delta_nabla_w;
	_delta_nabla_w.copy_size(this->w());

	std::tuple< gdw::nn*, gdw::layer*, gdw::layer* > _ctx = make_tuple(this, &_delta_nabla_b, &_delta_nabla_w);
	_fn(_args, zpt::context(&_ctx));
	return std::make_tuple(_delta_nabla_b, _delta_nabla_w);
}

void gdw::NNLayer::builtins() {
	try {
		zpt::lambda::add("gdw::nn::train::gradient_descent", 3,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::nn* _nn = (gdw::nn*) _ctx->unpack();
				zpt::json _training_set = _args[0];
				size_t _batch_size = (size_t) _args[1];
				size_t _epochs = (size_t) _args[2];

				std::cout << "Starting training for " << _epochs << " epochs, using " << _batch_size << " length batches from a " << _training_set->arr()->size() << " length training set, with a factor of " << _nn->learning_rate() << " for learning rate." << endl << flush;
				
				for (size_t _e = 0; _e != _epochs; _e++) {
					std::srand(std::time(nullptr));
					std::random_shuffle(_training_set->arr()->begin(), _training_set->arr()->end());
					std::vector< zpt::json > _batch(_training_set->arr()->begin(), _training_set->arr()->begin() + _batch_size);

					gdw::layer _nabla_b(1, _nn->b().n_cols);
					gdw::layer _nabla_w(1, _nn->w().n_cols);

					for (auto _x_y : _batch) {
						gdw::layer_delta _delta_nabla = _nn->back_propagate(_x_y[0], _x_y[1]);
						for (size_t _l = 1; _l != _nabla_b.n_cols; _l++) {
							if (_nabla_b(_l).n_elem == 0) {
								_nabla_b(_l) = std::get<0>(_delta_nabla)(_l);
							}
							else {
								_nabla_b(_l) = _nabla_b(_l) + std::get<0>(_delta_nabla)(_l);
							}
							if (_nabla_w(_l).n_elem == 0) {
								_nabla_w(_l) = std::get<1>(_delta_nabla)(_l);
							}
							else {
								_nabla_w(_l) = _nabla_w(_l) + std::get<1>(_delta_nabla)(_l);
							}
						}
					}

					for (size_t _l = 1; _l != _nn->b().n_cols; _l++) {
						_nn->b()(_l) = _nn->b()(_l) - ((_nn->learning_rate() / _batch.size()) * _nabla_b(_l));
						_nn->w()(_l) = _nn->w()(_l) - ((_nn->learning_rate() / _batch.size()) * _nabla_w(_l));
					}

					// std::cout << "Epoch: " << _e << "\r" << flush;
					//std::cout << "--------------------------------------------" << endl << _nn->w()(1) << endl << _nn->w()(2) << endl << endl << flush;
				}
				// std::cout << endl << flush;
				return zpt::undefined;
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}	
	try {
		zpt::lambda::add("gdw::nn::feed_forward::sigmoid", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::nn* _nn = (gdw::nn*) _ctx->unpack();
				zpt::json _x = _args[0];
				arma::mat _a = gdw::matrix::to_matrix(_x).t();

				// std::cout << "INPUT\n" << _a << endl << flush;
				// std::cout << "WEIGHTS\n" << (_nn->w()(1)) << endl << flush;
				// std::cout << "FIRST LAYER RESULT\n" << ((_nn->w()(1) * _a)) << endl << flush;
				
				for (size_t _l = 1; _l != _nn->b().n_cols; _l++) {
					_a = gdw::sigmoid((_nn->w()(_l) * _a) + _nn->b()(_l));
				}	

				return gdw::matrix::from_matrix(_a.t());
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::nn::back_propagate::gradient_descent", 2,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				std::tuple< gdw::nn*, gdw::layer*, gdw::layer* >* _params = (std::tuple< gdw::nn*, gdw::layer*, gdw::layer* >*) _ctx->unpack();
				gdw::nn* _nn = std::get<0>(*_params);
				size_t _n_layers = _nn->b().n_cols;
				gdw::layer* _nabla_b = std::get<1>(*_params);
				gdw::layer* _nabla_w = std::get<2>(*_params);
				arma::mat _x = gdw::matrix::to_matrix(_args[0]);
				arma::mat _y = gdw::matrix::to_matrix(_args[1]).t();
				arma::mat _a = _x.t();
				arma::mat _z;
				gdw::layer _zs(1, _n_layers);
				gdw::layer _as(1, _n_layers);
				_as(0) = _a;

				for (size_t _l = 1; _l != _n_layers; _l++) {
					_z = (_nn->w()(_l) * _a) + _nn->b()(_l);
					_a = gdw::sigmoid(_z);
					_zs(_l) = _z;
					_as(_l) = _a;
				}	

				arma::mat _delta = (_a - _y) % gdw::sigmoid_prime(_z);
				(*_nabla_b)(0, _n_layers - 1) = _delta;
				(*_nabla_w)(0, _n_layers - 1) = _delta * _as(0, _n_layers - 2).t();

				for(size_t _l = 2; _l != _n_layers; _l++) {
					_z = _zs(_n_layers - _l);
					arma::mat _sp = gdw::sigmoid_prime(_z);
					_delta = (_nn->w()(_n_layers - _l + 1).t() * _delta) % _sp;
					(*_nabla_b)(0, _n_layers - _l) = _delta;
					(*_nabla_w)(0, _n_layers - _l) = _delta * _as(0, _n_layers - _l - 1).t();
				}
				
				return zpt::undefined;
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}	
}

arma::mat gdw::sigmoid(arma::mat _z) {
	return 1 / (1 + arma::exp(-_z));
}

arma::mat gdw::sigmoid_prime(arma::mat _z) {
	return gdw::sigmoid(_z) % (1 - gdw::sigmoid(_z));
}
