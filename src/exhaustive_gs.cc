// @file:     exhaustive_gs.cc
// @author:   Samuel
// @created:  2020.01.30
// @license:  Apache License 2.0
//
// @desc:     Exhaustive ground state configuration finder for DB charges.

#include <stdlib.h>

#include "exhaustive_gs.h"

using namespace egs;

int egsglobal::log_level = Logger::WRN;
SimParams EGS::sim_params;
std::mutex EGS::result_store_mutex;
std::vector<Results> EGS::thread_results;

constexpr auto sp = &EGS::sim_params;

void SimParams::setDBLocs(const std::vector<DBLocEucl> &t_db_locs)
{
  db_locs = t_db_locs;
  if (db_locs.size() == 0) {
    throw "There must be 1 or more DBs when setting DBs for SimParams.";
  }
  n_dbs = db_locs.size();
  v_ext.resize(n_dbs);
  for (unsigned int i=0; i<v_ext.size(); i++)
    v_ext[i] = 0;
  db_r.resize(n_dbs, n_dbs);
  v_ij.resize(n_dbs, n_dbs);
}

void SimParams::setDBLocs(const std::vector<DBLocLC> &t_db_locs)
{
  std::vector<DBLocEucl> db_locs_eucl;
  for (DBLocLC db_loc_lc : t_db_locs)
    db_locs_eucl.push_back(latCoordToEucl(db_loc_lc[0], 
          db_loc_lc[1], db_loc_lc[2]));
  setDBLocs(db_locs_eucl);
}

EGS::EGS(SimParams &sp)
{
  sim_params = sp;
  initialize();
}

void EGS::storeResults(EGSThread *finder, int thread_id)
{
  result_store_mutex.lock();

  thread_results[thread_id] = finder->threadResults();

  result_store_mutex.unlock();
}

void EGS::initialize()
{
  Logger log(egsglobal::log_level);

  log.debug() << "Performing pre-calculations..." << std::endl;

  // phys
  FPType Kc = 1/(4 * consts::PI * sp->eps_r * consts::EPS0);

  // return the Euclidean distance between DBs i and j
  auto d_eucl = [](const int &i, const int &j) -> FPType {
    FPType x1 = sim_params.db_locs[i].first;
    FPType y1 = sim_params.db_locs[i].second;
    FPType x2 = sim_params.db_locs[j].first;
    FPType y2 = sim_params.db_locs[j].second;
    return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
  };

  auto coulomb_pot = [Kc](const FPType &r) -> FPType {
    return consts::Q0 * Kc * exp(-r/(sim_params.debye*1e-9)) / r;
  };

  // inter-db distances and voltages
  for (int i=0; i<sp->n_dbs; i++) {
    sp->db_r(i,i) = 0.;
    sp->v_ij(i,i) = 0.;
    for (int j=i+1; j<sp->n_dbs; j++) {
      sp->db_r(i,j) = 1e-10 * d_eucl(i,j); // convert angstrom to m
      sp->v_ij(i,j) = coulomb_pot(sp->db_r(i,j));
      sp->db_r(j,i) = sp->db_r(i,j);
      sp->v_ij(j,i) = sp->v_ij(i,j);

      log.debug() << "db_r[" << i << "][" << j << "]=" << sp->db_r(i,j) 
        << ", v_ij[" << i << "][" << j << "]=" << sp->v_ij(i,j) << std::endl;
    }
  }

  log.debug() << "Pre-calculations complete" << std::endl << std::endl;
}

void EGS::invoke()
{
  Logger log(egsglobal::log_level);
  log.debug() << "Setting up EGSThreads..." << std::endl;

  // determine threading
  int max_config_id = pow(sp->base, sp->db_locs.size());
  int cpu_thread_count = std::thread::hardware_concurrency();
  if (sp->num_threads > cpu_thread_count) {
    log.warning() << "Specified thread count: " << sp->num_threads <<
      " is higher than available CPU threads: " << cpu_thread_count << std::endl;
  }
  if (sp->num_threads <= 0 || sp->num_threads > max_config_id) {
    sp->num_threads = std::min(cpu_thread_count, max_config_id);
  }
  thread_results.resize(sp->num_threads);
  int step_size = sp->num_threads;

  // spawn all the threads
  for (int i=0; i<sim_params.num_threads; i++) {
    EGSThread finder(i, i, step_size);
    std::thread th(&EGSThread::invoke, finder);
    egs_threads.push_back(std::move(th));
  }

  log.debug() << "Wait for simulations to complete..." << std::endl;

  // wait for threads to complete
  for (auto &th : egs_threads) {
    th.join();
  }

  log.debug() << "All simulations complete." << std::endl;

  // find the actual ground states among the returned states
  bool gs_init = false;
  FPType gs_energy;
  for (Results t_results : thread_results) {
    for (Result result : t_results) {
      if (!gs_init) {
        gs_energy = result.energy;
        filtered_results.push_back(result);
        gs_init = true;
      } else if (sp->scope != GroundStates 
          || std::abs(result.energy - gs_energy) < consts::ZERO_EQUIV) {
        filtered_results.push_back(result);
      } else if (sp->scope == GroundStates && result.energy < gs_energy) {
        filtered_results.clear();
        filtered_results.push_back(result);
        gs_energy = result.energy;
      }
    }
  }
}




EGSThread::EGSThread(const int &thread_id, const int &ind, const int &step_size)
  : thread_id(thread_id), step_size(step_size), 
    config(ind, sp->db_locs.size(), sp->base)
{
}

EGSThread::~EGSThread()
{
}

void EGSThread::invoke()
{
  Logger log(egsglobal::log_level);

  bool gs_init=false;
  FPType gs_energy;
  bool has_next = true;

  auto eq = [](FPType a, FPType b) -> bool {
    return std::abs(a - b) <= consts::ZERO_EQUIV;
  };
  auto less_than = [](FPType a, FPType b) -> bool {
    return (b - a) > consts::ZERO_EQUIV;
  };

  auto qubo_routine = [this, &gs_init, &gs_energy, eq, less_than]() {
    FPType energy = config.systemEnergy();
    if (sp->scope == AllStates || (gs_init && eq(energy, gs_energy))) {
      results.push_back(Result(config.config(), energy, false));
    } else if (!gs_init) {
      gs_init = true;
      gs_energy = energy;
      results.push_back(Result(config.config(), energy, false));
    } else {
      if (less_than(energy, gs_energy)) {
        gs_energy = energy;
        results.clear();
        results.push_back(Result(config.config(), energy, false));
      }
    }
  };

  auto gs_model_routine = [this, &gs_init, &gs_energy, eq, less_than]() {
    bool stable = config.isMetastable();
    if (sp->scope == AllStates || (stable && sp->scope == ValidStates)) {
      results.push_back(Result(config.config(), config.systemEnergy(), stable));
    } else if (stable) {
      FPType energy = config.systemEnergy();
      if (!gs_init) {
        gs_init = true;
        gs_energy = energy;
        results.push_back(Result(config.config(), energy, stable));
      } else if (less_than(energy, gs_energy)) {
        results.clear();
        results.push_back(Result(config.config(), energy, stable));
        gs_energy = energy;
      } else if (eq(energy, gs_energy)) {
        results.push_back(Result(config.config(), energy, stable));
      }
    }
  };

  while (has_next) {
    if (sp->qubo) {
      qubo_routine();
    } else {
      gs_model_routine();
    }

    if (!config.inc(step_size)) {
      has_next = false;
    }
  }

  EGS::storeResults(this, thread_id);
}




void ChgConfig::setConfig(IntVec t_config)
{
  chg_config.resize(t_config.size());
  std::copy(t_config.begin(), t_config.end(), chg_config.begin());
  chg_index = configToIndex(chg_config, base);
  v_local.resize(chg_config.size());
  calc_dirty = true;
  max_index = pow(base, chg_config.size()) - 1;
}

IntVec ChgConfig::indexToConfig(int index, const int &n_dbs, const int &base)
{
  IntVec vec(n_dbs);
  for (int i=0; i<n_dbs; i++)
    vec[i] = -1;

  int i=n_dbs - 1;
  while (index > 0) {
    div_t d;
    d = div(index, base);
    index = d.quot;
    vec[i--] = d.rem - 1;
  }
  return vec;
}

int ChgConfig::configToIndex(const IntVec &config, const int &base)
{
  int index = 0;
  for (unsigned int i=0; i<config.size(); i++) {
    index += (config[i] + 1) * pow(base, config.size() - i - 1);
  }
  return index;
}

std::string ChgConfig::configToStr(const IntVec &config)
{
  std::string config_str;
  for (unsigned int i=0; i<config.size(); i++) {
    switch(config[i]) {
      case -1:
        config_str.push_back('-');
        break;
      case +1:
        config_str.push_back('+');
        break;
      case 0:
        config_str.push_back('0');
        break;
      default:
        throw std::invalid_argument("Unknown charge state encountered: " + std::to_string(config[i]));
    }
  }
  return config_str;
}

bool ChgConfig::inc(const int &steps)
{
  Logger log(egsglobal::log_level);
  if (chg_index + steps > max_index) {
     log.debug() << "Step size causes index to exceed max index." << std::endl;
     return false;
  }

  calc_dirty = true;
  chg_index += steps;
  chg_config = indexToConfig(chg_index, chg_config.size(), base);

  /*
  int carry = steps;
  int i = chg_config.size() - 1;
  int max_chg = base - 2;
  while (carry > 0 && i >= 0) {
    chg_config[i] += carry;
    if (chg_config[i] > max_chg) {
      carry = chg_config[i] - max_chg;
      chg_config[i] = max_chg;
    } else {
      carry = 0;
    }
    i--;
  }
  */
  
  return true;
}

void ChgConfig::calculateLocalPotential(const int &i)
{
  ublas::matrix_column<FPMat> v_i(sp->v_ij, i);
  v_local[i] = - sp->v_ext[i] - ublas::inner_prod(v_i, chg_config);
}

FPType ChgConfig::systemEnergy()
{
  if (!sp->qubo) {
    // ground state model calculation
    if (calc_dirty) {
      v_local = - sp->v_ext - ublas::prod(sp->v_ij, chg_config);
      /* TODO test whether this is faster
      for (unsigned int i=0; i<chg_config.size(); i++) {
        calculateLocalPotential(i);
      }
      */
      calc_dirty = false;
    }
    return 0.5 * ublas::inner_prod(sp->v_ext, chg_config) -
      0.5 * ublas::inner_prod(chg_config, v_local);
  } else {
    // QUBO mapping calculation
    return ublas::inner_prod(sp->v_ext - ublas::vector<FPType>(chg_config.size(),sp->muzm), chg_config)
      + 0.5 * ublas::inner_prod(chg_config, ublas::prod(sp->v_ij, chg_config));
  }
}

FPVec ChgConfig::localPotentials()
{
  if (calc_dirty) {
    v_local = - sp->v_ext - ublas::prod(sp->v_ij, chg_config);
    calc_dirty = false;
  }
  return v_local;
}

bool ChgConfig::isMetastable()
{
  Logger log(egsglobal::log_level);

  const FPType &muzm = sp->muzm;
  const FPType &mupz = sp->muzm - consts::MUDIFF;
  const FPType &zero_equiv = consts::ZERO_EQUIV;

  for (unsigned int i=0; i<chg_config.size(); i++) {
    calculateLocalPotential(i);

    // return false if invalid
    int chg = chg_config[i];
    if (!(   (chg == -1 && v_local[i] + muzm < zero_equiv)    // DB-
          || (chg == 1  && v_local[i] + mupz > - zero_equiv)  // DB+
          || (chg == 0  && v_local[i] + muzm > - zero_equiv   // DB0
                            && v_local[i] + mupz < zero_equiv))) {
      log.debug() << "config " << configToStr(chg_config) 
        << " has an invalid population, failed at index " << i << std::endl;
      log.debug() << "v_local[i]=" << v_local[i] << ", muzm=" << muzm 
        << ", mupz=" << mupz << std::endl;
      return false;
    }
  }
  log.debug() << "config " << configToStr(chg_config) 
    << " has a valid population." << std::endl;

  auto hopDel = [this](const int &i, const int &j) -> FPType {
    int dn_i = (chg_config[i]==-1) ? 1 : -1;
    int dn_j = - dn_i;
    return - v_local[i]*dn_i - v_local[j]*dn_j - sp->v_ij(i,j);
  };

  for (unsigned int i=0; i<chg_config.size(); i++) {
    // do nothing with DB+
    if (chg_config[i] == 1)
      continue;

    for (unsigned int j=0; j<chg_config.size(); j++) {
      // attempt hops from more negative charge states to more positive ones
      FPType E_del = hopDel(i, j);
      if ((chg_config[j] > chg_config[i]) && (E_del < -zero_equiv)) {
        log.debug() << "config " << configToStr(chg_config) 
          << " not stable since hopping from site " << i << " to " << j 
          << " would result in an energy change of " << E_del << std::endl;
        return false;
      }
    }
  }
  log.debug() << "config " << configToStr(chg_config) 
    << " has a stable configuration." << std::endl;
  return true;
}
