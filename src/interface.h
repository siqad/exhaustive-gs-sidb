// @file:     interface.h
// @author:   Samuel
// @created:  2020.01.31
// @license:  Apache License 2.0
//
// @desc:     Simulation interface which manages reads and writes with 
//            SiQADConn as well as invokes EGS instances.

#include "siqadconn.h"
#include "exhaustive_gs.h"

namespace egs {

  namespace bpt = boost::property_tree;

  typedef std::pair<FPType,FPType> EuclCoord2D;

  class EGSInterface
  {
  public:
    //! Constructure for SimAnnealInterface. Set defer_var_loading to true if
    //! you don't want simulation parameters to be loaded immediately from
    //! SiQADConn.
    EGSInterface(std::string t_in_path, std::string t_out_path, 
        std::string t_ext_pots_path, int t_ext_pots_step);

    ~EGSInterface();

    //! Read external potentials.
    FPVec loadExternalPotentials(const int &n_dbs);

    //! Prepare simulation variables.
    SimParams loadSimParams(bool qubo);

    //! Write the simulation results to output file.
    void writeSimResults();


    //! Run the simulation, returns 0 if simulation was successful.
    int invoke(SimParams sparams);

  private:

    // Instances
    phys::SiQADConnector *sqconn=nullptr;
    EGS *master_finder=nullptr;

    // variables
    std::string in_path;
    std::string out_path;
    std::string ext_pots_path;
    int ext_pots_step;

  };
}
