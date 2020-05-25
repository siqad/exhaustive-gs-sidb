// @file:     interface.cc
// @author:   Samuel
// @created:  2020.01.31
// @license:  Apache License 2.0
//
// @desc:     Simulation interface which manages reads and writes with 
//            SiQADConn as well as invokes EGS instances.

#include "interface.h"

// std
#include <iterator>
#include <algorithm>
#include <string>

// boost
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace egs;

EGSInterface::EGSInterface(std::string t_in_path, 
                                       std::string t_out_path,
                                       std::string t_ext_pots_path,
                                       int t_ext_pots_step)
  : in_path(t_in_path), out_path(t_out_path), ext_pots_path(t_ext_pots_path),
    ext_pots_step(t_ext_pots_step)
{
  sqconn = new phys::SiQADConnector(std::string("ExhaustiveGS"), in_path, out_path);
  //loadSimParams();
}

EGSInterface::~EGSInterface()
{
  delete master_finder;
  delete sqconn;
}

ublas::vector<FPType> EGSInterface::loadExternalPotentials(const int &n_dbs)
{
  Logger log(egsglobal::log_level);
  bpt::ptree pt;
  bpt::read_json(ext_pots_path, pt);

  const bpt::ptree &pot_steps_arr = pt.get_child("pots");
  // iterate pots array until the desired step has been reached
  if (static_cast<unsigned long>(ext_pots_step) >= pot_steps_arr.size())
    throw std::range_error("External potential step out of bounds.");
  bpt::ptree::const_iterator pots_arr_it = std::next(pot_steps_arr.begin(), 
      ext_pots_step);

  ublas::vector<FPType> v_ext;
  v_ext.resize(n_dbs);
  int db_i = 0;
  for (bpt::ptree::value_type const &v : (*pots_arr_it).second) {
    log.debug() << "Reading v_ext[" << db_i << "]" << std::endl;
    v_ext[db_i] = (v.second.get_value<FPType>());
    log.debug() << "v_ext[" << db_i << "] = " << v_ext[db_i] << std::endl;
    db_i++;
  }
  return v_ext;
}

SimParams EGSInterface::loadSimParams(bool qubo)
{
  Logger log(egsglobal::log_level);

  auto sqparam = [this](std::string key) {return sqconn->getParameter(key);};

  SimParams sp;

  sp.qubo = qubo;

  // grab all physical locations
  log.debug() << "Grab all physical locations..." << std::endl;
  std::vector<DBLocEucl> db_locs;
  for(auto db : *(sqconn->dbCollection())) {
    db_locs.push_back(SimParams::latCoordToEucl(db->n, db->m, db->l));
    log.debug() << "DB loc: x=" << db_locs.back().first
        << ", y=" << db_locs.back().second << std::endl;
  }
  sp.setDBLocs(db_locs);

  // load external voltages if relevant file has been supplied
  if (!ext_pots_path.empty()) {
    log.debug() << "Loading external potentials..." << std::endl;
    sp.v_ext = loadExternalPotentials(sp.db_locs.size());
  } else {
    log.debug() << "No external potentials file supplied, set to 0." << std::endl;
    for (auto &v : sp.v_ext) {
      v = 0;
    }
  }


  // VAIRABLE INITIALIZATION
  log.debug() << "Retrieving variables from SiQADConn..." << std::endl;

  sp.autofail = std::stoi(sqparam("autofail"));
  sp.num_threads = std::stoi(sqparam("num_threads"));
  sp.base = qubo ? 2 : std::stoi(sqparam("base"));
  if (sqparam("scope") == "ground") {
    sp.scope = GroundStates;
  } else if (sqparam("scope") == "valid") {
    sp.scope = ValidStates;
  } else if (sqparam("scope") == "all") {
    sp.scope = AllStates;
  } else {
    throw std::invalid_argument("Unknown simulation scope encountered.");
  }

  sp.muzm = std::stod(sqparam("muzm"));
  sp.debye = std::stod(sqparam("debye"));
  sp.eps_r = std::stod(sqparam("eps_r"));

  log.debug() << "Retrieval from SiQADConn complete." << std::endl;

  return sp;
}

void EGSInterface::writeSimResults()
{
  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(EGS::sim_params.db_locs.size());
  for (unsigned int i = 0; i < EGS::sim_params.db_locs.size(); i++) { //need the index
    dbl_data[i].first = std::to_string(EGS::sim_params.db_locs[i].first);
    dbl_data[i].second = std::to_string(EGS::sim_params.db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  std::vector<std::vector<std::string>> db_dist_data;
  for (Result result : master_finder->filteredResults()) {
    std::vector<std::string> db_dist;
    std::string config_str = ChgConfig::configToStr(result.config);
    if (EGS::sim_params.qubo)
      std::replace(config_str.begin(), config_str.end(), '-', '1');
    db_dist.push_back(config_str); // config
    db_dist.push_back(std::to_string(result.energy));         // energy
    db_dist.push_back(std::to_string(1));                     // occurance freq
    db_dist.push_back(std::to_string(result.stable));         // metastability
    db_dist.push_back(std::to_string(EGS::sim_params.base));  // 3-state
    db_dist_data.push_back(db_dist);
  }
  sqconn->setExport("db_charge", db_dist_data);

  sqconn->writeResultsXml();
}

int EGSInterface::invoke(SimParams sparams)
{
  master_finder = new EGS(sparams);
  master_finder->invoke();
  return 0;
}
