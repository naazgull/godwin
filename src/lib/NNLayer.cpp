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

gdw::neural_net::neural_net(gdw::nn* _target) : std::shared_ptr< gdw::NNLayer >(_target) {
}

gdw::neural_net::~neural_net() {
}

gdw::neural_net gdw::neural_net::unpack(zpt::context _ctx) {
	return gdw::neural_net((gdw::nn*) _ctx->unpack());
}

std::string gdw::NNLayer::__matrix_names[4] = { "weights", "outputs", "deltas", "sigmas" };

gdw::NNLayer::NNLayer() {
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		this->__matrix.push_back(gdw::mat_ptr());
	}
	try {
		zpt::lambda::add("gdw::nn::linear", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::neural_net _nn = gdw::neural_net::unpack(_ctx);
				gdw::index_t _node = (gdw::index_t) _args[0];

				arma::colvec _weights = _nn->matrix(WEIGHTS)->col(_node);
				std::cout << _weights << endl << flush;
				arma::rowvec _outputs = _nn->matrix(OUTPUTS)->row(0);
				std::cout << _outputs << endl << flush;
				double _product = arma::dot(_outputs, _weights);
				std::cout << _product << endl << flush;
				return zpt::json::floating(_product);
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::nn::sigmoid", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				return zpt::json::floating(1.0 / (double) (1.0 + exp(-_args[0]->dbl())));
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::nn::stochastic_gradient_descent", 4,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::neural_net _nn = gdw::neural_net::unpack(_ctx);
				gdw::index_t _node_idx = (gdw::index_t) _args[0];
				double _expected = (double) _args[1];
				zpt::json _node = _nn->network()["nodes"][_node_idx];

				double _dn = 0;
				double _sn = 0;
				double _on = 0;//(double) _network["layers"][_layer][_node]["output"];
				double _tn = 0;//(double) _expected_output[_node];
				
				if ((bool) _node["is_output"]) {
					_sn = _on * (1 - _on) * (_tn - _on);
					//_network["layers"][_layer][_node] << "sigma" << _sn;
					_dn = 0.05 * _sn;
				}
				else {
					double _snk = 0;
					/*for (auto _nk : _network["layers"][_layer + 1]->arr()) {
						_snk += ((double) _network["layers"]);
						}*/
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
	if (!this->__network->ok()) {
		this->__network = zpt::json::object();
	}
	if (!this->__network["defaults"]->ok()) {
		this->__network << "defaults" << zpt::json::object();
	}
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__network["defaults"]["lambdas"] << "value" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_value_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "value" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_threshold_lambda(zpt::lambda _function) {
	if (!this->__network->ok()) {
		this->__network = zpt::json::object();
	}
	if (!this->__network["defaults"]->ok()) {
		this->__network << "defaults" << zpt::json::object();
	}
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__network["defaults"]["lambdas"] << "threshold" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_threshold_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "threshold" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_backpropagation_lambda(zpt::lambda _function) {
	if (!this->__network->ok()) {
		this->__network = zpt::json::object();
	}
	if (!this->__network["defaults"]->ok()) {
		this->__network << "defaults" << zpt::json::object();
	}
	if (!this->__network["defaults"]["lambdas"]->ok()) {
		this->__network["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__network["defaults"]["lambdas"] << "backpropagation" << zpt::json::lambda(_function);
}

void gdw::NNLayer::set_backpropagation_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__network["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__network["nodes"][_neuron]["lambdas"]->ok()) {
		this->__network["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__network["nodes"][_neuron]["lambdas"] << "backpropagation" << zpt::json::lambda(_function);
}

gdw::index_t gdw::NNLayer::push(zpt::json _neuron, size_t _n_neurons) {
	if (!this->__network->ok()) {
		this->__network = zpt::json::object();
	}
	if (!this->__network["nodes"]->ok()) {
		this->__network << "nodes" << zpt::json::array();
	}
	const size_t _position = this->__network["nodes"]->arr()->size();
	for (size_t _n = 0; _n != _n_neurons; _n++) {
		this->__network["nodes"] << _neuron;
	}
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		this->__matrix[_m]->resize((_m == WEIGHTS ? this->__matrix[_m]->n_rows + _n_neurons : 1), this->__matrix[_m]->n_cols + _n_neurons);
		this->__matrix[_m]->submat((_m == WEIGHTS ? _position : 0), _position, (_m == WEIGHTS ? this->__matrix[_m]->n_rows - 1 : 0), this->__matrix[_m]->n_cols - 1).zeros();
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
	zpt::json _return = zpt::json::array();
	gdw::index_t _node_idx = 0;
	gdw::index_t _n = 0;
	arma::uvec _next;
	
	for (auto _node : this->__network["nodes"]->arr()) {
		if ((bool) _node["is_input"]) {
			(*this->__matrix[OUTPUTS])(0, _node_idx) = _input[_n];
			_n++;
			_next = arma::unique(arma::join_cols(_next, arma::find(this->__matrix[WEIGHTS]->row(_node_idx))));
		}
		_node_idx++;
	}

	return this->classify(_next);
	/*	else {
			zpt::json _node = this->__network["nodes"][_node_idx];
			zpt::json _args({ _node_idx });
			double _value = (
				_node["lambdas"]["value"]->type() == zpt::JSLambda ?
				_node["lambdas"]["value"]->lbd()->call(_args, zpt::context(this)) :
				this->__network["defaults"]["lambdas"]["value"]->lbd()->call(_args, zpt::context(this)));

			zpt::json _threshold_args({ _value });
			double _output = (double) (
				_node["lambdas"]["threshold"]->type() == zpt::JSLambda ?
				_node["lambdas"]["threshold"]->lbd()->call(_threshold_args, zpt::context(this)) :
				this->__network["defaults"]["lambdas"]["threshold"]->lbd()->call(_threshold_args, zpt::context(this)));

			(*this->__matrix[OUTPUTS])(0, _node_idx) = _output;
			if ((bool) _node["is_output"]) {
				_return << _value;
			}
		}
		_node_idx++;
		}*/
}

zpt::json gdw::NNLayer::classify(arma::uvec& _to_process) {
	zpt::json _return = zpt::json::array();
	arma::uvec _next;

	for (auto _node_idx : _to_process) {
		zpt::json _node = this->__network["nodes"][(size_t) _node_idx];
		zpt::json _args({ _node_idx });

		double _value = (
			_node["lambdas"]["value"]->type() == zpt::JSLambda ?
			_node["lambdas"]["value"]->lbd()->call(_args, zpt::context(this)) :
			this->__network["defaults"]["lambdas"]["value"]->lbd()->call(_args, zpt::context(this))
		);
		
		zpt::json _threshold_args({ _value });
		double _output = (double) (
			_node["lambdas"]["threshold"]->type() == zpt::JSLambda ?
			_node["lambdas"]["threshold"]->lbd()->call(_threshold_args, zpt::context(this)) :
			this->__network["defaults"]["lambdas"]["threshold"]->lbd()->call(_threshold_args, zpt::context(this))
		);
		
		(*this->__matrix[OUTPUTS])(0, _node_idx) = _output;
		
		if ((bool) _node["is_output"]) {
			_return << _value;
		}
		else {
			_next = arma::unique(arma::join_cols(_next, arma::find(this->__matrix[WEIGHTS]->row(_node_idx))));
		}
	}
	
	if (_next.size() != 0) {
		_return = _return + this->classify(_next);
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


