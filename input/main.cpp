#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

using namespace std;

string trim(const string& str,
		    const string& whitespace = " \t")
{
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos) return "";
	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;
	return str.substr(strBegin, strRange);
}



int main() {

	string input_file = "./obstacles_in_flow.dat";
	ifstream infile(input_file);

	string s;
	int line_num = 0;

	double domainx;

	vector<double> samp_time;


	while (std::getline(infile, s)) {
		line_num++;
        istringstream iss(s);
		vector<string> tokens{istream_iterator<string>{iss},istream_iterator<string>{}};

		if (tokens.size() <= 0) continue; //skip empty line

		tokens[0] = trim(tokens[0]);
		if (tokens[0].substr(0,2) == "//") continue; //skip comment line

		transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);

		if (tokens[0] == "domain_size_x") {
			domainx = stod(tokens[1]);
			continue;
		}

		if (tokens[0] == "sampling_time") {
			for (int i=1; i < tokens.size(); i++)
				samp_time.push_back(stod(tokens[i]));
			continue;
		}		
	}
	infile.close();

	cout << "\nDomain size x = " << domainx << endl;

	cout << "\nSample time = ";
	for (int i =0; i<samp_time.size(); i++)
		cout << samp_time[i] << "   ";
    cout << endl;

	return 0;
}
