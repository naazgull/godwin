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
#include <godwin/Matrix.h>

gdw::mat_ptr::mat_ptr() : std::shared_ptr< arma::mat >(new arma::mat()) {
}

gdw::mat_ptr::mat_ptr(const arma::uword _in_rows, const arma::uword _in_cols) : std::shared_ptr< arma::mat >(new arma::mat(_in_rows, _in_cols)) {
}

gdw::mat_ptr::mat_ptr(const arma::SizeMat& _size) : std::shared_ptr< arma::mat >(new arma::mat(_size)) {
}

gdw::mat_ptr::mat_ptr(const char* _text) : std::shared_ptr< arma::mat >(new arma::mat(_text)) {
}

gdw::mat_ptr::mat_ptr(const std::string& _text) : std::shared_ptr< arma::mat >(new arma::mat(_text)) {
}

gdw::mat_ptr::mat_ptr(const std::vector<double>& _vector) : std::shared_ptr< arma::mat >(new arma::mat(_vector)) {
}

gdw::mat_ptr::mat_ptr(const std::initializer_list<double>& _list) : std::shared_ptr< arma::mat >(new arma::mat(_list)) {
}

gdw::mat_ptr::mat_ptr(const std::initializer_list< std::initializer_list<double> >& _list) : std::shared_ptr< arma::mat >(new arma::mat(_list)) {
}

gdw::mat_ptr::mat_ptr(arma::mat&& _other) : std::shared_ptr< arma::mat >(new arma::mat(_other)) {
}

gdw::mat_ptr::mat_ptr(const arma::mat& _other) : std::shared_ptr< arma::mat >(new arma::mat(_other)) {
}

gdw::mat_ptr::~mat_ptr() {
}

