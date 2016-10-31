/*
Copyright (c) 2016
*/

#include <godwin/godwin.h>

class CollectionListener : public zpt::EventListener {
public:
	inline CollectionListener() : zpt::EventListener(zpt::rest::url_pattern({ "{api-version}", "{collection-name}" })) {
	}
	inline virtual ~CollectionListener() {
	}

	inline virtual zpt::json get(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		zpt::json _list;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _list = _db->query({collection-name}, _envelope["payload"]);

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _list = _db->query({collection-name}, (!_envelope["payload"]->ok() || _envelope["payload"]->obj()->size() == 0 ? std::string("*") : std::string("*") + ((std::string) _envelope["payload"]->obj()->begin()->second) + std::string("*")));
		
		if (!_list->ok()) {
		 	return { "status", 204 };
		}
		return {
			"status", 200,
			"payload", _list
		};
	}

	inline virtual zpt::json post(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		assertz(
			_envelope["payload"]->ok() &&
			_envelope["payload"]["a"]->ok() &&
			_envelope["payload"]["b"]->ok() &&
			_envelope["payload"]["b"]->ok(),
			"required fields: 'a', 'b' and 'c'", 412, 0);
		assertz(
			_envelope["payload"]["c"]->type() == zpt::JSArray,
			"invalid field type: 'c' must be a list of strings", 400, 0);

		std::string _id;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _id = _db->insert({collection-name}, _resource, _envelope["payload"]);

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _id = _db->insert({collection-name}, _resource, _envelope["payload"]);

		std::string _href = (_resource + (_resource.back() != '/' ? std::string("/") : std::string("")) + _id);					
		
		return {
			"status", 200,
			"payload", {
				"id", _id,
				"href", _href
			}
		};
	}

	inline virtual zpt::json head(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		zpt::json _list;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _list = _db->query({collection-name}, _envelope["payload"]);

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _list = _db->query({collection-name}, (!_envelope["payload"]->ok() || _envelope["payload"]->obj()->size() == 0 ? std::string("*") : std::string("*") + ((std::string) _envelope["payload"]->obj()->begin()->second) + std::string("*")));
		
		if (!_list->ok()) {
		 	return { "status", 204 };
		}
		return {
			"status", 200,
			"headers", {
				"Content-Length", ((std::string) _list).length()
			}
		};
	}
};

class DocumentListener : public zpt::EventListener {
public:
	inline DocumentListener() : zpt::EventListener(zpt::rest::url_pattern({ "{api-version}", "{collection-name}", "([^/]+)" })) {
	}
	inline virtual ~DocumentListener() {
	}

	inline virtual zpt::json get(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		zpt::json _document;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _envelope["payload"] << "_id" << _resource;
		// _document = _db->query({collection-name}, _envelope["payload"]);
		// if (!_document->ok() || _document["size"] == 0) {
		// 	return { "status", 404 };
		// }
		// _document = _document["elements"][0];

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _document = _db->get({collection-name}, _resource);
		// if (!_document->ok()) {
		//	return { "status", 404 };
		// }
		
		return {
			"status", 200,
			"payload", _document
		};
	}

	inline virtual zpt::json put(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		assertz(
			_envelope["payload"]->ok() &&
			_envelope["payload"]["a"]->ok() &&
			_envelope["payload"]["b"]->ok() &&
			_envelope["payload"]["b"]->ok(),
			"required fields: 'a', 'b' and 'c'", 412, 0);
		assertz(
			_envelope["payload"]["c"]->type() == zpt::JSArray,
			"invalid field type: 'c' must be a list of strings", 400, 0);

		size_t _size = 0;
	
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _size = _db->save({collection-name}, { "_id", _resource }, _envelope["payload"]);

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _size = _db->save({collection-name}, _resource, _envelope["payload"]);
	
		return {
			"status", 200,
			"payload", {
				"updated", _size
			}
		};
	}

	inline virtual zpt::json head(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		zpt::json _document;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _envelope["payload"] << "_id" << _resource;
		// _document = _db->query({collection-name}, _envelope["payload"]);
		// if (!_document->ok() || _document["size"] == 0) {
		// 	return { "status", 404 };
		// }
		// _document = _document["elements"][0];

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _document = _db->get({collection-name}, _resource);
		// if (!_document->ok()) {
		//	return { "status", 404 };
		// }
		
		return {
			"status", 200,
			"headers", {
				"Content-Length", ((std::string) _document).length()
			}
		};
	}

	inline virtual zpt::json del(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		size_t _size = 0;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _size = _db->remove({collection-name}, { "_id", _resource });

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _size = _db->remove({collection-name}, _resource);

		return {
			"status", 200,
			"payload", {
				"removed", _size
			}
		};
	}

	inline virtual zpt::json head(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) 
		size_t _size = 0;
		
		// Typical MongoDB pattern
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();
		// _size = _db->set({collection-name}, { "_id", _resource }, _envelope["payload"]);

		// Typical Redis pattern
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();
		// _size = _db->set({collection-name}, _resource, _envelope["payload"]);

		return {
			"status", 200,
			"payload", {
				"updated", _size
			}
		};	
	}
};

class ControllerListener : public zpt::EventListener {
public:
	inline ControllerListener() : zpt::EventListener(zpt::rest::url_pattern({ "{api-version}", "{contoller-name}" }) {
	}
	inline virtual ~ControllerListener() {
	}

	inline virtual zpt::json post(std::string _resource, zpt::json _envelope, zpt::EventEmitterPtr _emitter) {
		// Get MongoDB connection
		// zpt::mongodb::Client* _db = (zpt::mongodb::Client*) _emitter->get_kb("mongodb.godwin").get();

		// Get Redis connection
		// zpt::redis::Client* _db = (zpt::redis::Client*) _emitter->get_kb("redis.godwin").get();

		return {
			"status", 200,
			"payload", {
				"text", "some response"
			}
		};
	}
};

extern "C" void restify(zpt::EventEmitterPtr _emitter) {
	// Setup a MongoDB connection
	// zpt::KBPtr _kb(new zpt::mongodb::Client(_emitter->options(), "mongodb.godwin"));
	// _emitter->add_kb("mongodb.godwin", _kb);
	
	// Setup a Redis connection
	// zpt::KBPtr _kb(new zpt::redis::Client(_emitter->options(), "redis.godwin"));
	// _emitter->add_kb("redis.godwin", _kb);
	
	_emitter->on(zpt::ev::listener(new CollectionListener()));
	_emitter->on(zpt::ev::listener(new DocumentListener()));
	_emitter->on(zpt::ev::listener(new ControllerListener()));
}

