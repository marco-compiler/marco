//
// Created by Ale on 21/06/2021.
//

#include "../include/modelica/utils/VariableFilter.h"

void modelica::VariableFilter::dump() {
    for (int s = 0; s < 12; ++s) printf("#");
    cout << "\n *** TRACKED VARIABLES *** : " << endl;

    for (const auto &vt : _variables) {
        vt.dump();
    }
    for (int s = 0; s < 12; ++s) printf("#");
    cout << "\n *** TRACKED REGEX(s) *** : " << endl;
    for (const auto &regex : _regex) {
        printf("Regex: /%s/", regex.c_str());
    }

}

void modelica::VariableFilter::addRegexString(string regex) {
    _bypass = false; //turnoff bypass by default
    _regex.push_back(regex);
}

void modelica::VariableFilter::addVariable(VariableTracker var) {
    _bypass = false; //turnoff bypass by default
    _variables.push_back(var);
}
bool modelica::VariableFilter::isBypass() const { return _bypass; }
void modelica::VariableFilter::setBypass(bool bypass) { _bypass = bypass; }
bool modelica::VariableFilter::matchesRegex(const string &identifier)
{
	for (const auto &regexString : _regex) {
		std::regex regex(regexString, std::regex::ECMAScript);
		if (regex_match(identifier, regex)) {
			return true;
		}
	}
	return false;
}
VariableTracker modelica::VariableFilter::lookupByIdentifier(
		const string &identifier)
{
	if (std::any_of(
			_variables.begin(),
			_variables.end(),
			[&identifier](VariableTracker i) {
				return i.getName().compare(identifier);
			})) {
		for (const auto &varTracker : _variables) {
			if (varTracker.getName() == (identifier))
				return varTracker;
		}
	}
}
bool modelica::VariableFilter::checkTrackedIdentifier(const string &identifier)
{
	for (const auto &varTracker : _variables) {
		if (varTracker.getName() == (identifier))
			return true;
	}
	return false;
}
