#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <string>
#include <unordered_map>
#include <iostream>


class Config {
public:
//	static Config& getInstance(int argc, char** argv)
//	{
//		Config cfg (argc, argv);
//		static Config& instance = cfg;
//		return instance;
//	}
	Config(int argc, char** argv);

	void print_help();

	void add_params();

	std::string get_param_value(std::string);

private:
	struct Param {
		std::string des;
		std::string val;
	};
	std::unordered_map<std::string, Param> params;

public:
	Config() = delete;
	Config(Config const&) = delete;
	void operator=(Config const&) = delete;
};
#endif
