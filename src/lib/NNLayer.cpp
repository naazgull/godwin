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
#include <godwin/NNLayer.h>
#include <random>

gdw::neural_net::neural_net() : std::shared_ptr< gdw::NNLayer >(new gdw::NNLayer()) {
}

gdw::neural_net::~neural_net() {
}

gdw::NNLayer::NNLayer(){
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		this->__matrix.push_back(gdw::mat_ptr());
	}
	try {
		zpt::lambda::add("gdw::nn::linear", 2,
			[] (zpt::json _args, unsigned short _nargs) -> zpt::json {
				size_t _layer = (size_t) _args[0];
				size_t _node = (size_t) _args[1];
				zpt::json _network = _args[1];

				double _value = 0;
				for (size_t _i = 0; _i != _node["inbound"]->arr()->size(); _i++) {
					zpt::json _inbound_node = _network["layers"][((size_t) _node["inbound"][_i][0])][((size_t) _node["inbound"][_i][1])];
					_value += ((double) _node["weigths"][_i]) * ((double) _inbound_node["output"]);
				}
				return zpt::json::floating(_value);
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::nn::sigmoid", 1,
			[] (zpt::json _args, unsigned short _nargs) -> zpt::json {
				return zpt::json::floating(1.0 / (double) (1.0 + exp(-_args[0]->dbl())));
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::nn::stochastic_gradient_descent", 4,
			[] (zpt::json _args, unsigned short _nargs) -> zpt::json {
				size_t _layer = (size_t) _args[0];
				size_t _node = (size_t) _args[1];
				zpt::json _expected_output = _args[2];
				zpt::json _network = _args[3];
				size_t _last = _network["layers"]->arr()->size() - 1;

				double _dn = 0;
				double _sn = 0;
				double _on = (double) _network["layers"][_layer][_node]["output"];
				double _tn = (double) _expected_output[_node];
				
				if (_layer == _last) {
					_sn = _on * (1 - _on) * (_tn - _on);
					_network["layers"][_layer][_node] << "sigma" << _sn;
					_dn = 0.05 * _sn;
				}
				else {
					double _snk = 0;
					for (auto _nk : _network["layers"][_layer + 1]->arr()) {
						_snk += ((double) _network["layers"]);
					}
					_sn = _on * (1 - _on) * _snk;
				}
				
				return zpt::undefined;
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}	
}

gdw::NNLayer::~NNLayer() {
}

zpt::json gdw::NNLayer::network() {
	return this->__network;
}

gdw::mat_ptr gdw::NNLayer::matrix(gdw::index_t _which) {
	return this->__matrix[_which];
}

void gdw::NNLayer::set_value_lambda(zpt::lambda _function) {
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::mkobj();
	}
	this->__network["defaults"]["lambdas"] << "value" << _function;
}

void gdw::NNLayer::set_value_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::mkobj();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "value" << _function;
}

void gdw::NNLayer::set_threshold_lambda(zpt::lambda _function) {
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::mkobj();
	}
	this->__network["defaults"]["lambdas"] << "threshold" << _function;
}

void gdw::NNLayer::set_threshold_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::mkobj();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "threshold" << _function;
}

void gdw::NNLayer::set_backpropagation_lambda(zpt::lambda _function) {
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::mkobj();
	}
	this->__network["defaults"]["lambdas"] << "backpropagation" << _function;
}

void gdw::NNLayer::set_backpropagation_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::mkobj();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "backpropagation" << _function;
}

gdw::index_t gdw::NNLayer::push(gdw::index_t _layer, size_t _n_neurons) {
	const size_t _position = this->__network["nodes"]->arr()->size();
	for (size_t _n = 0; _n != _n_neurons; _n++) {
		this->__network["layers"][_layer] << this->__network["nodes"]->arr()->size();
		this->__network["nodes"] << zpt::mkobj();
	}
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		this->__matrix[_m]->resize(this->__matrix[_m]->n_rows + _n_neurons, (_m == WEIGHTS ? this->__matrix[_m]->n_cols + _n_neurons : 1));
		this->__matrix[_m]->submat(_position, (_m == WEIGHTS ? _position : 0), this->__matrix[_m]->n_rows - 1, (_m == WEIGHTS ? this->__matrix[_m]->n_cols - 1 : 0)).zeros();
	}
	return _position;
}

void gdw::NNLayer::wire(gdw::index_t _neuron, zpt::json _inbound) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	assertz(_inbound->type() == zpt::JSArray, "source node list '_inbound' must be a JSON array", 412, 0);

	std::random_device _rd;
	std::mt19937 _gen(_rd());
	std::uniform_real_distribution<double> _dis(-0.05, 0.05);		
	for (auto _source : _inbound->arr()) {
		(*this->__matrix[WEIGHTS])(((gdw::index_t) _source), _neuron) =  _dis(_gen);
	}
}

void gdw::NNLayer::wire(zpt::json _network) {
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__network = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__network[gdw::NNLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__network[gdw::NNLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

void gdw::NNLayer::wire(std::string _network_serialized) {
	zpt::json _network(_network_serialized);
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__network = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__network[gdw::NNLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__network[gdw::NNLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

void gdw::NNLayer::wire(std::istream _input_stream) {
	zpt::json _network;
	_input_stream >> _network;
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__network = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__network[gdw::NNLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__network[gdw::NNLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

std::string gdw::NNLayer::snapshot() {
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		std::ostringstream _oss;
		this->__matrix[_m]->save(_oss, arma::arma_binary);
		std::string _matrix_str(_oss.str());
		zpt::base64::encode(_matrix_str);
		this->__network << gdw::NNLayer::__matrix_names[_m] << _matrix_str;
	}
	return std::string(this->__network);
}

void gdw::NNLayer::snapshot(std::ostream _output_stream) {
	_output_stream << this->snapshot() << flush;
}

void gdw::NNLayer::train(zpt::json _input, zpt::json _expected_output) {
	zpt::json _classification = this->classify(_input);
	this->adjust(_input, _expected_output, _classification);
}

zpt::json gdw::NNLayer::classify(zpt::json _input) {
	zpt::json _return = zpt::mkarr();
	
	size_t _last = this->__network["layers"]->arr()->size() - 1;
	size_t _idx = 0;

	for (auto _layer : this->__network["layers"]->arr()) {
		if (_idx == 0) {
			for (size_t _n = 0; _n != _layer->arr()->size(); _n++) {
				_layer[_n] << "value" << _input[_n] << "output" << _input[_n];
			}
		}
		else {
			for (auto _node : _layer->arr()) {
				zpt::json _args({ _node, this->__network });
				double _value = (_node["lambdas"]["value"]->type() == zpt::JSLambda ? _node["lambdas"]["value"]->lbd()->call(_args) : this->__network["defaults"]["lambdas"]["value"]->lbd()->call(_args));

				zpt::json _threshold_args({ _value });
				double _output = (double) (_node["lambdas"]["threshold"]->type() == zpt::JSLambda ? _node["lambdas"]["threshold"]->lbd()->call(_threshold_args) : this->__network["defaults"]["lambdas"]["threshold"]->lbd()->call(_threshold_args));
				
				_node << "value" << _value << "output" << _output;
				if ( _idx == _last) {
					_return << _value;
				}
			}
		}
		_idx++;
	}
	return _return;
}

void gdw::NNLayer::adjust(zpt::json _input, zpt::json _expected_output, zpt::json _output) {
	size_t _last = this->__network["layers"]->arr()->size() - 1;

	for (size_t _idx = _last; _idx != 0; _idx--) {		
		zpt::json _layer = this->__network["layers"][_idx];
		for (size_t _n = 0; _n != _layer->arr()->size(); _n++) {
			zpt::json _node = _layer[_n];
			zpt::json _args({ _idx, _n, _expected_output, this->__network });
			double _delta = 0;
		}
	}
	
}

 void gdw::NNLayer::adjust(size_t _layer, size_t _neuron, double weight) {
}

