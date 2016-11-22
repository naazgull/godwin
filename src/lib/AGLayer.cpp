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
#include <godwin/AGLayer.h>
#include <random>

gdw::acyclic_graph::acyclic_graph() : std::shared_ptr< gdw::AGLayer >(new gdw::AGLayer()) {
}

gdw::acyclic_graph::acyclic_graph(zpt::json _sizes) : std::shared_ptr< gdw::AGLayer >(new gdw::AGLayer(_sizes)) {
}

gdw::acyclic_graph::acyclic_graph(gdw::ag* _target) : std::shared_ptr< gdw::AGLayer >(_target) {
}

gdw::acyclic_graph::~acyclic_graph() {
}

std::string gdw::AGLayer::__matrix_names[N_MATRIX] = { "weights", "outputs", "deltas", "sigmas", "units" };

gdw::AGLayer::AGLayer() : __learning_rate(0.05), __weight_random_limits{ 0.01, 0.5 } {
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		this->__matrix.push_back(gdw::mat_ptr());
	}
	this->builtins();
	this->set_value_lambda(zpt::lambda("gdw::ag::linear", 1));
	this->set_threshold_lambda(zpt::lambda("gdw::ag::sigmoid", 1));
	this->set_error_delta_lambda(zpt::lambda("gdw::ag::gradient_descent::deltas", 2));
	this->set_weight_adjust_lambda(zpt::lambda("gdw::ag::gradient_descent::weights", 1));
}

gdw::AGLayer::AGLayer(zpt::json _sizes) : gdw::AGLayer() {
	assertz(_sizes->type() == zpt::JSArray, "'_sizes' parameter is not a valid array", 412, 0);
	size_t _l = 0;
	std::vector< std::tuple< gdw::index_t, size_t > > _limits;
	for (auto _len : _sizes->arr()) {
		assertz(_len->type() == zpt::JSInteger, "'_sizes' parameter is not a valid array", 412, 0);
		_limits.push_back(make_tuple(this->push((_l == 0 ? zpt::json({ "is_input", true }) : (_l == _sizes->arr()->size() - 1 ? zpt::json({ "is_output", true }) : zpt::json::object() ) ), (size_t) _len), (size_t) _len));
		_l++;
	}

	zpt::json _upstream;
	for (auto _l_limit : _limits) {
		zpt::json _this_layer = zpt::json::array();
		for ( size_t _i = std::get<0>(_l_limit); _i != std::get<0>(_l_limit) + std::get<1>(_l_limit); _i++) {
			_this_layer << _i;
			if (_upstream->ok()) {
				this->wire(_i, _upstream);
			}
		}
		_upstream = _this_layer;
	}
}

gdw::AGLayer::~AGLayer() {
}

zpt::json gdw::AGLayer::graph() {
	return this->__graph;
}

double gdw::AGLayer::learning_rate() {
	return this->__learning_rate;
}

void gdw::AGLayer::learning_rate(double _learning_rate) {
	this->__learning_rate = _learning_rate;
}

double* gdw::AGLayer::weight_generation_limits() {
	return this->__weight_random_limits;
}

void gdw::AGLayer::weight_generation_limits(double _lower, double _higher) {
	this->__weight_random_limits[0] = _lower;
	this->__weight_random_limits[1] = _higher;
}

gdw::mat_ptr gdw::AGLayer::matrix(gdw::index_t _which) {
	return this->__matrix[_which];
}

void gdw::AGLayer::set_value_lambda(zpt::lambda _function) {
	if (!this->__graph->ok()) {
		this->__graph = zpt::json::object();
	}
	if (!this->__graph["defaults"]->ok()) {
		this->__graph << "defaults" << zpt::json::object();
	}
	if (!this->__graph["defaults"]["lambdas"]->ok()) {
		this->__graph["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__graph["defaults"]["lambdas"] << "value" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_value_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__graph["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__graph["nodes"][_neuron]["lambdas"]->ok()) {
		this->__graph["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__graph["nodes"][_neuron]["lambdas"] << "value" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_threshold_lambda(zpt::lambda _function) {
	if (!this->__graph->ok()) {
		this->__graph = zpt::json::object();
	}
	if (!this->__graph["defaults"]->ok()) {
		this->__graph << "defaults" << zpt::json::object();
	}
	if (!this->__graph["defaults"]["lambdas"]->ok()) {
		this->__graph["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__graph["defaults"]["lambdas"] << "threshold" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_threshold_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__graph["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__graph["nodes"][_neuron]["lambdas"]->ok()) {
		this->__graph["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__graph["nodes"][_neuron]["lambdas"] << "threshold" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_error_delta_lambda(zpt::lambda _function) {
	if (!this->__graph->ok()) {
		this->__graph = zpt::json::object();
	}
	if (!this->__graph["defaults"]->ok()) {
		this->__graph << "defaults" << zpt::json::object();
	}
	if (!this->__graph["defaults"]["lambdas"]->ok()) {
		this->__graph["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__graph["defaults"]["lambdas"] << "deltas" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_error_delta_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__graph["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__graph["nodes"][_neuron]["lambdas"]->ok()) {
		this->__graph["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__graph["nodes"][_neuron]["lambdas"] << "deltas" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_weight_adjust_lambda(zpt::lambda _function) {
	if (!this->__graph->ok()) {
		this->__graph = zpt::json::object();
	}
	if (!this->__graph["defaults"]->ok()) {
		this->__graph << "defaults" << zpt::json::object();
	}
	if (!this->__graph["defaults"]["lambdas"]->ok()) {
		this->__graph["defaults"] << "lambdas" << zpt::json::object();
	}
	this->__graph["defaults"]["lambdas"] << "weights" << zpt::json::lambda(_function);
}

void gdw::AGLayer::set_weight_adjust_lambda(gdw::index_t _neuron, zpt::lambda _function) {
	assertz(this->__graph["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	if (!this->__graph["nodes"][_neuron]["lambdas"]->ok()) {
		this->__graph["nodes"][_neuron] << "lambdas" << zpt::json::object();
	}
	this->__graph["nodes"][_neuron]["lambdas"] << "weights" << zpt::json::lambda(_function);
}

gdw::index_t gdw::AGLayer::push(zpt::json _neuron, size_t _n_neurons) {
	if (!this->__graph->ok()) {
		this->__graph = zpt::json::object();
	}
	if (!this->__graph["nodes"]->ok()) {
		this->__graph << "nodes" << zpt::json::array();
	}
	const size_t _position = this->__graph["nodes"]->arr()->size();
	for (size_t _n = 0; _n != _n_neurons; _n++) {
		this->__graph["nodes"] << _neuron;
	}
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		bool _square_matrix = (_m == WEIGHTS || _m == DELTAS);
		this->__matrix[_m]->resize((_square_matrix ? this->__matrix[_m]->n_rows + _n_neurons : 1), this->__matrix[_m]->n_cols + _n_neurons);
		this->__matrix[_m]->submat((_square_matrix ? _position : 0), _position, (_square_matrix ? this->__matrix[_m]->n_rows - 1 : 0), this->__matrix[_m]->n_cols - 1).zeros();
	}
	return _position;
}

void gdw::AGLayer::wire(gdw::index_t _neuron, zpt::json _inbound) {
	assertz(this->__graph["nodes"][_neuron]->ok(), "neuron index does not match any existant", 412, 0);
	assertz(_inbound->type() == zpt::JSArray, "source node list '_inbound' must be a JSON array", 412, 0);
	
	std::random_device _rd;
	std::mt19937 _gen(_rd());
	//std::uniform_real_distribution<double> _dis(this->__weight_random_limits[0], this->__weight_random_limits[1]);
	for (auto _source : _inbound->arr()) {
		double _weight = 0;
		do {
			//_weight = _dis(_gen);
			_weight = this->__weight_random_limits[0] + std::generate_canonical<double, 10>(_gen) * std::abs(this->__weight_random_limits[1] - this->__weight_random_limits[0]);
		}
		while(_weight == 0);
		(*this->__matrix[WEIGHTS])(((gdw::index_t) _source), _neuron) = _weight;
	}
}

void gdw::AGLayer::wire(zpt::json _network) {
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__graph = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__graph[gdw::AGLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__graph[gdw::AGLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

void gdw::AGLayer::wire(std::string _network_serialized) {
	zpt::json _network(_network_serialized);
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__graph = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__graph[gdw::AGLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__graph[gdw::AGLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

void gdw::AGLayer::wire(std::istream _input_stream) {
	zpt::json _network;
	_input_stream >> _network;
	assertz(_network->type() == zpt::JSObject && _network["layers"]->type() == zpt::JSArray, "provided JSON object not in a recognizable format", 412, 0);
	this->__graph = _network;
	
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		if (this->__graph[gdw::AGLayer::__matrix_names[_m]]->type() == zpt::JSString) {
			std::string _matrix_str(this->__graph[gdw::AGLayer::__matrix_names[_m]]->str());
			zpt::base64::decode(_matrix_str);
			std::istringstream _iss(_matrix_str);
			this->__matrix[_m]->load(_iss, arma::arma_binary);
		}

	}
}

std::string gdw::AGLayer::snapshot() {
	for (size_t _m = 0; _m != N_MATRIX; _m++) {
		std::ostringstream _oss;
		this->__matrix[_m]->save(_oss, arma::arma_binary);
		std::string _matrix_str(_oss.str());
		zpt::base64::encode(_matrix_str);
		this->__graph << gdw::AGLayer::__matrix_names[_m] << _matrix_str;
	}
	return std::string(this->__graph);
}

void gdw::AGLayer::snapshot(std::ostream _output_stream) {
	_output_stream << this->snapshot() << flush;
}

zpt::json gdw::AGLayer::train(zpt::json _input, zpt::json _expected_output) {
	zpt::json _classification = this->classify(_input);
	this->calculate_deltas(_input, _expected_output);
	this->adjust_weights(_input);
	return _classification;
}

zpt::json gdw::AGLayer::classify(zpt::json _input) {
	zpt::json _return = zpt::json::array();
	gdw::index_t _node_idx = 0;
	gdw::index_t _n = 0;
	arma::uvec _downstream;
	
	for (auto _node : this->__graph["nodes"]->arr()) {
		if ((bool) _node["is_input"]) {
			(*this->__matrix[UNITS])(0, _node_idx) = _input[_n];
			(*this->__matrix[OUTPUTS])(0, _node_idx) = _input[_n];
			_n++;
			_downstream = arma::unique(arma::join_cols(_downstream, arma::find(this->__matrix[WEIGHTS]->row(_node_idx))));
		}
		_node_idx++;
	}

	if (_downstream.size() != 0) {
		zpt::json _return = this->classify(_downstream);
		//std::cout << "[" << this->matrix(OUTPUTS)->row(0) << "]" << endl << flush;
		return _return;
	}
	return zpt::undefined;
}

zpt::json gdw::AGLayer::classify(arma::uvec& _to_process) {
	zpt::json _return = zpt::json::array();
	arma::uvec _downstream;

	for (auto _node_idx : _to_process) {
		zpt::json _node = this->__graph["nodes"][(size_t) _node_idx];
		
		zpt::json _value_args({ _node_idx });
		zpt::lambda _fn_net = (_node["lambdas"]["value"]->type() == zpt::JSLambda ? _node["lambdas"]["value"]->lbd() : this->__graph["defaults"]["lambdas"]["value"]->lbd());
		zpt::json _value = _fn_net(_value_args, zpt::context(this));

		(*this->__matrix[UNITS])(0, _node_idx) = _value;
		
		if ((bool) _node["is_output"]) {
			(*this->__matrix[OUTPUTS])(0, _node_idx) = _value;
			_return << _value;
		}
		else {
			zpt::json _threshold_args({ _value });
			zpt::lambda _fn_output = (_node["lambdas"]["threshold"]->type() == zpt::JSLambda ?_node["lambdas"]["threshold"]->lbd() : this->__graph["defaults"]["lambdas"]["threshold"]->lbd());
			double _output = (double) _fn_output(_threshold_args, zpt::context(this));
		
			(*this->__matrix[OUTPUTS])(0, _node_idx) = _output;
			
			_downstream = arma::unique(arma::join_cols(_downstream, arma::find(this->__matrix[WEIGHTS]->row(_node_idx))));
		}
	}
	
	if (_downstream.size() != 0) {
		_return = _return + this->classify(_downstream);
	}
	return _return;
}

void gdw::AGLayer::calculate_deltas(zpt::json _input, zpt::json _expected_output) {
	gdw::index_t _node_idx = 0;
	gdw::index_t _n = 0;
	arma::uvec _upstream;
	
	for (auto _node : this->__graph["nodes"]->arr()) {
		if ((bool) _node["is_output"]) {
			zpt::json _error_args({ _node_idx, _expected_output[_n] });
			zpt::lambda _fn_error = (_node["lambdas"]["deltas"]->type() == zpt::JSLambda ? _node["lambdas"]["deltas"]->lbd() : this->__graph["defaults"]["lambdas"]["deltas"]->lbd());
			zpt::json _value = _fn_error(_error_args, zpt::context(this));
			_n++;
			_upstream = arma::unique(arma::join_cols(_upstream, arma::find(this->__matrix[WEIGHTS]->col(_node_idx))));			
		}
		_node_idx++;
	}

	if (_upstream.size() != 0) {
		this->calculate_deltas(_upstream);
	}
}

void gdw::AGLayer::calculate_deltas(arma::uvec& _to_process) {
	arma::uvec _upstream;
	
	for (gdw::index_t _node_idx : _to_process) {
		zpt::json _node = this->__graph["nodes"][_node_idx];
		zpt::json _error_args({ _node_idx, 0 });
		zpt::lambda _fn_error = (_node["lambdas"]["deltas"]->type() == zpt::JSLambda ? _node["lambdas"]["deltas"]->lbd() : this->__graph["defaults"]["lambdas"]["deltas"]->lbd());
		zpt::json _value = _fn_error(_error_args, zpt::context(this));
		if (!((bool) _node["is_input"])) {
			_upstream = arma::unique(arma::join_cols(_upstream, arma::find(this->__matrix[WEIGHTS]->col(_node_idx))));
		}
	}

	if (_upstream.size() != 0) {
		this->calculate_deltas(_upstream);
	}
}

void gdw::AGLayer::adjust_weights(zpt::json _input) {
	gdw::index_t _node_idx = 0;
	arma::uvec _upstream;
	
	for (auto _node : this->__graph["nodes"]->arr()) {
		if ((bool) _node["is_output"]) {
			zpt::json _weight_args({ _node_idx });
			zpt::lambda _fn_weights = (_node["lambdas"]["weights"]->type() == zpt::JSLambda ? _node["lambdas"]["weights"]->lbd() : this->__graph["defaults"]["lambdas"]["weights"]->lbd());
			_fn_weights(_weight_args, zpt::context(this));

			_upstream = arma::unique(arma::join_cols(_upstream, arma::find(this->__matrix[WEIGHTS]->col(_node_idx))));			
		}
		_node_idx++;
	}

	if (_upstream.size() != 0) {
		this->adjust_weights(_upstream);
	}
}

void gdw::AGLayer::adjust_weights(arma::uvec& _to_process) {
	arma::uvec _upstream;
	
	for (gdw::index_t _node_idx : _to_process) {
		zpt::json _node = this->__graph["nodes"][_node_idx];

		zpt::json _weight_args({ _node_idx });
		zpt::lambda _fn_weights = (_node["lambdas"]["weights"]->type() == zpt::JSLambda ? _node["lambdas"]["weights"]->lbd() : this->__graph["defaults"]["lambdas"]["weights"]->lbd());
		_fn_weights(_weight_args, zpt::context(this));

		if (!((bool) _node["is_input"])) {
			_upstream = arma::unique(arma::join_cols(_upstream, arma::find(this->__matrix[WEIGHTS]->col(_node_idx))));
		}
	}

	if (_upstream.size() != 0) {
		this->adjust_weights(_upstream);
	}
}

void gdw::AGLayer::builtins() {
	try {
		zpt::lambda::add("gdw::ag::linear", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::ag* _nn = (gdw::ag*) _ctx->unpack();
				gdw::index_t _node_idx = (gdw::index_t) _args[0];
				std::cout << _nn->matrix(WEIGHTS)->col(_node_idx) << " ø " << endl << _nn->matrix(OUTPUTS)->row(0) << " = " << arma::dot(_nn->matrix(WEIGHTS)->col(_node_idx), _nn->matrix(OUTPUTS)->row(0)) << endl << flush;
				return zpt::json::floating(arma::dot(_nn->matrix(WEIGHTS)->col(_node_idx), _nn->matrix(OUTPUTS)->row(0)));
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::ag::sigmoid", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				return zpt::json::floating(1.0 / (double) (1.0 + std::exp(-_args[0]->dbl())));
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}
	try {
		zpt::lambda::add("gdw::ag::gradient_descent::deltas", 2,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::ag* _nn = (gdw::ag*) _ctx->unpack();
				gdw::index_t _node_idx = (gdw::index_t) _args[0];
				zpt::json _node = _nn->graph()["nodes"][_node_idx];

				double _sn = 0;
				double _on = (*_nn->matrix(OUTPUTS))(0, _node_idx);
				
				if ((bool) _node["is_output"]) {
					double _tn = (double) _args[1];
					_sn = _on * (1 - _on) * (_on - _tn);
					(*_nn->matrix(SIGMAS))(0, _node_idx) = _sn;
				}
				else {
					arma::uvec _downstream = arma::find(_nn->matrix(WEIGHTS)->row(_node_idx));
					double _snk = 0;
					for (auto _nk : _downstream) {
						_snk += (*_nn->matrix(WEIGHTS))(_node_idx, _nk) * (*_nn->matrix(SIGMAS))(0, _nk);
					}
					_sn = _on * (1 - _on) * _snk;
					(*_nn->matrix(SIGMAS))(0, _node_idx) = _sn;
				}

				return zpt::json::floating(_sn);
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}	
	try {
		zpt::lambda::add("gdw::ag::gradient_descent::weights", 1,
			[] (zpt::json _args, unsigned short _nargs, zpt::context _ctx) -> zpt::json {
				gdw::ag* _nn = (gdw::ag*) _ctx->unpack();
				gdw::index_t _node_idx = (gdw::index_t) _args[0];
				zpt::json _node = _nn->graph()["nodes"][_node_idx];

				if (!((bool) _node["is_input"])) {
					arma::uvec _upstream = arma::find(_nn->matrix(WEIGHTS)->col(_node_idx));
					for (auto _nk : _upstream) {
						(*_nn->matrix(DELTAS))(_nk, _node_idx) = _nn->learning_rate() * (*_nn->matrix(OUTPUTS))(0, _node_idx) * (*_nn->matrix(SIGMAS))(0, _node_idx);
						if ((*_nn->matrix(WEIGHTS))(_nk, _node_idx) == (*_nn->matrix(DELTAS))(_nk, _node_idx)) {
							(*_nn->matrix(DELTAS))(_nk, _node_idx) += _nn->learning_rate() * (*_nn->matrix(SIGMAS))(0, _node_idx);
						}
						(*_nn->matrix(WEIGHTS))(_nk, _node_idx) = (*_nn->matrix(WEIGHTS))(_nk, _node_idx) - (*_nn->matrix(DELTAS))(_nk, _node_idx);
					}
				}
				
				return zpt::undefined;
			}
		);
	}
	catch(zpt::AssertionException& _e) {
	}	
}
