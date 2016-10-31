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

gdw::neural_net::neural_net() : std::shared_ptr< gdw::NNLayer >(new gdw::NNLayer()) {
}

gdw::neural_net::~neural_net() {
}

gdw::NNLayer::NNLayer(){
}

gdw::NNLayer::~NNLayer() {
}

zpt::json gdw::NNLayer::network() {
	return this->__network;
}

void gdw::NNLayer::set_value_lambda(zpt::lambda _function) {
}

void gdw::NNLayer::set_value_lambda(size_t _layer, size_t _neuron, zpt::lambda _function) {
}

void gdw::NNLayer::set_threshold_lambda(zpt::lambda _function) {
}

void gdw::NNLayer::set_threshold_lambda(size_t _layer, size_t _neuron, zpt::lambda _function) {
}

void gdw::NNLayer::set_backpropagation_lambda(zpt::lambda _function) {
}

void gdw::NNLayer::set_backpropagation_lambda(size_t _layer, size_t _neuron, zpt::lambda _function) {
}

void gdw::NNLayer::push(size_t _layer) {
}

void gdw::NNLayer::wire(size_t _layer, size_t _neuron, zpt::json _inbound) {
}

void gdw::NNLayer::wire(zpt::json _network) {
}

void gdw::NNLayer::wire(std::string _network_serialized) {
}

void gdw::NNLayer::wire(std::istream _input_stream) {
}

std::string gdw::NNLayer::snapshot() {
	return std::string(this->__network);
}

void gdw::NNLayer::snapshot(std::ostream _output_stream) {
}

void gdw::NNLayer::train(zpt::json _input, zpt::json _expected_output) {
}

zpt::json gdw::NNLayer::classify(zpt::json _input) {
	return zpt::undefined;
}

void gdw::NNLayer::adjust(size_t _layer, size_t _neuron, double weight) {
}

