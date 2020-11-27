// @file:     exhaustive_gs.h
// @author:   Samuel
// @created:  2020.01.30
// @license:  Apache License 2.0
//
// @desc:     Exhaustive ground state configuration finder for DB charges.

#include <vector>
#include <mutex>
#include <thread>
#include <iostream>
#include <exception>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>


#include "global.h"

namespace consts{
  // lattice
  const FPType lat_a = 3.84;  // lattice vector in x, angstroms (intra dimer row)
  const FPType lat_b = 7.68;  // lattice vector in y, angstroms (inter dimer row)
  const FPType lat_c = 2.25;  // dimer pair separation, angstroms

  // physics
  const FPType Q0 = 1.602E-19;
  const FPType Kb = 8.617E-5;
  const FPType EPS0 = 8.854E-12;
  const FPType MUDIFF = 0.59;
  const FPType PI = 3.14159;

  // simulation

  // Allowed headroom in eV for physically invalid configurations to still be 
  // considered valid.
  const FPType ZERO_EQUIV = 1E-6;
}

namespace egs {
  namespace ublas = boost::numeric::ublas;

  // forward declaration
  struct Result;
  class ChgConfig;
  class EGSThread;

  typedef ublas::vector<int> IntVec;
  typedef ublas::vector<FPType> FPVec;
  typedef ublas::matrix<FPType> FPMat;

  typedef std::pair<FPType,FPType> DBLocEucl;
  typedef std::vector<int> DBLocLC;
  typedef std::vector<Result> Results;

  enum ResultScope { GroundStates, ValidStates, AllStates };

  //! Simulation parameters.
  class SimParams
  {
  public:

    //! Empty constructor
    SimParams() {};

    //! Constructor taking DBs in Euclidean coordinates.
    SimParams(const std::vector<DBLocEucl> &t_db_locs) {setDBLocs(t_db_locs);}

    //! Constructor taking DBs in Lattice coordinates. They are internally 
    //! converted to Euclidean coordinates.
    SimParams(const std::vector<DBLocLC> &t_db_locs) {setDBLocs(t_db_locs);}

    //! Set DB locations given Euclideans coordinates.
    void setDBLocs(const std::vector<DBLocEucl> &t_db_locs);

    //! Set DB locations given Lattice coordinates. They are internally 
    //! converted to Euclidean coordinates.
    void setDBLocs(const std::vector<DBLocLC> &t_db_locs);

    //! Convert lattice coordinates (n, m, l) to a pair of Euclidean coordinates 
    //! in angstrom.
    static DBLocEucl latCoordToEucl(int n, int m, int l)
    {
      FPType x = n * consts::lat_a;
      FPType y = m * consts::lat_b + l * consts::lat_c;
      return std::make_pair(x, y);
    }


    // VARIABLES
    bool qubo=false;      // true for qubo mapping, false for normal gs model
    int num_threads=-1;
    int base=3;           // 3-state or 2-state search
    int autofail=16;      // autofail DB count
    ResultScope scope=GroundStates;

    std::vector<DBLocEucl> db_locs; // List of DB locations
    FPType muzm=-0.25;    // mu_- in SiQAD paper (eV)
    FPType debye=5;       // Thomas-Fermi screening length (nm)
    FPType eps_r=5.6;     // Relative permittivity
    FPVec v_ext;          // External voltages

    // calculated / inferred from inputs
    int n_dbs;            // Number of DBs
    FPMat db_r;           // Inter-DB distances
    FPMat v_ij;           // Inter-DB Coulombic potentials
  };

  //! Single result.
  struct Result
  {
    Result(IntVec config, FPType energy, bool stable)
      : config(config), energy(energy), stable(stable) {};
    IntVec config;
    FPType energy;
    bool stable;
  };

  //! Main exhaustive ground state controller which spawns EGSThreads.
  class EGS
  {
  public:
    //! Constructor which takes SimParams and invokes pre-calculations.
    EGS(SimParams &sparams);

    //! Invoke EGS threads.
    void invoke();

    //! Store results from EGSThread.
    static void storeResults(EGSThread *finder, int thread_id);

    //! Retrieve filtered results.
    Results filteredResults() {return filtered_results;}

    //! Publicly accessible simulation parameters.
    static SimParams sim_params;

  private:
    //! Perform pre-calculations.
    void initialize();

    // VARIABLES
    static std::mutex result_store_mutex;   //! Thread mutex for result storage
    std::vector<std::thread> egs_threads;   //! Spawned threads
    static std::vector<Results> thread_results; //! Results from all threads
    Results filtered_results;               //! Filtered results after applying ResultScope
  };



  //! Charge configuration.
  class ChgConfig
  {
  public:
    //! Constructor using a configuration vector.
    ChgConfig(IntVec config, const int &base) : base(base)
    {
      setConfig(config);
    }

    //! Constructor using an index.
    ChgConfig(const int &index, const int &n_dbs, const int &base) : base(base)
    {
      setConfig(index, n_dbs);
    }

    //! Set config to another config.
    void setConfig(IntVec t_config);

    //! Set config from index.
    void setConfig(const int &t_index, const int &n_dbs)
    {
      setConfig(indexToConfig(t_index, n_dbs, base));
    }

    //! Return the config.
    IntVec config() const {return chg_config;}

    //! Return the index.
    int index() const {return chg_index;}

    //! Convert index to configuration vector.
    static IntVec indexToConfig(int index, const int &n_dbs, const int &base);

    //! Convert configuration vector to index.
    static int configToIndex(const IntVec &config, const int &base);

    //! Convert configuration vector to string.
    static std::string configToStr(const IntVec &config);

    //! Increment index by the specified number of steps, return whether 
    //! increment is successful.
    bool inc(const int &steps);

    //! Calculate v_local for the provided index.
    void calculateLocalPotential(const int &i);

    //! Return the system energy of the current configuration.
    FPType systemEnergy();

    //! Return the local potentials of the current configuration.
    FPVec localPotentials();

    //! Return whether the current configuration is metastable.
    bool isMetastable();


  private:

    
    bool calc_dirty;      // whether the current calculated values are dirty
    IntVec chg_config;    // charge configuration
    FPVec v_local;        // local potentials
    int chg_index;        // the index of this config
    int base;             // charge state count (2 for {DB-,DB0}, 3 to also include DB+)
    int max_index;        // maximum possible index given the charge count
  };



  //! Single EGS thread.
  class EGSThread
  {
  public:
    //! Constructor.
    EGSThread(const int &thread_id, const int &ind, const int &step_size);

    //! Destructor.
    ~EGSThread();

    //! Invoke simulation on this thread.
    void invoke();

    //! Return the results of this thread.
    Results threadResults() {return results;}

  private:
    int thread_id;
    int step_size;

    ChgConfig config;
    Results results;
  };

}
