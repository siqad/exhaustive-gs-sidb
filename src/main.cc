// @file:     main.cc
// @author:   Samuel
// @created:  2020.01.31
// @license:  Apache License 2.0
//
// @desc:     Main function for ExhaustiveGS

#include "global.h"
#include "interface.h"
#include <string>

using namespace egs;

int main(int argc, char *argv[])
{
  std::cout << "Physeng invoked" << std::endl;
  std::string if_name, of_name, ext_pots_name;
  std::vector<std::string> cml_args;

  std::cout << "*** Argument Parsing ***" << std::endl;

  if (argc < 3) {
    throw "Less arguments than excepted.";
  } else {
    // argv[0] is the binary
    if_name = argv[1];
    of_name = argv[2];

    // store the rest of the arguments
    for (int i=3; i<argc; i++) {
      cml_args.push_back(argv[i]);
    }
  }

  // parse additional arguments
  int ext_pots_step=0;
  bool qubo=false;
  unsigned long cml_i=0;
  while (cml_i < cml_args.size()) {
    if (cml_args[cml_i] == "--ext-pots") {
      std::cout << "--ext-pots: Import external potentials." << std::endl;
      ext_pots_name = cml_args[++cml_i];
    } else if (cml_args[cml_i] == "--ext-pots-step") {
      std::cout << "--ext-pots-step: Specify the step to use for potentials."
        << std::endl;
      ext_pots_step = stoi(cml_args[++cml_i]);
    } else if (cml_args[cml_i] == "--debug") {
      // show additional debug information
      std::cout << "--debug: Showing additional outputs." << std::endl;
      egsglobal::log_level = Logger::DBG;
    } else if (cml_args[cml_i] == "--qubo") {
      // use QUBO ground state energy
      std::cout << "--qubo: Use QUBO mapping, disables metastability checks." << std::endl;
      qubo = true;
    } else {
      throw "Unrecognized command-line argument: " + cml_args[cml_i];
    }
    cml_i++;
  }

  Logger log(egsglobal::log_level);

  log.echo() << "In File: " << if_name << std::endl;
  log.echo() << "Out File: " << of_name << std::endl;
  log.echo() << "External Potentials File: " << ext_pots_name << std::endl;

  log.echo() << "\n*** Initiate SimAnneal interface ***" << std::endl;
  EGSInterface interface(if_name, of_name, ext_pots_name, ext_pots_step);

  log.echo() << "\n*** Read Simulation parameters ***" << std::endl;
  SimParams sparams = interface.loadSimParams(qubo);

  if (sparams.n_dbs > sparams.autofail) {
    log.warning() << "Problem size > autofail threshold, exiting." << std::endl;
    return 1;
  }

  log.echo() << "\n*** Invoke simulation ***" << std::endl;
  interface.invoke(sparams);

  log.echo() << "\n*** Write simulation results ***" << std::endl;
  interface.writeSimResults();

  log.echo() << "\n*** SimAnneal Complete ***" << std::endl;
}
