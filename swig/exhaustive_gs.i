// @editted:    2020-01-31
// @license:    Apache License 2.0
//
// @desc:       Python wrapper for ExhaustiveGS

%module exhaustive_gs
%include <std_streambuf.i>
%include <std_sstream.i>
%include <std_iostream.i>
%include <std_deque.i>
%include <std_pair.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_map.i>
%include <exception.i>

namespace boost {
    namespace numeric {
        namespace ublas {
        }
    }
}

namespace egsglobal {
    extern int log_level;
}

extern int egsglobal::log_level;

%{
#include "logger.h"
#include "global.h"
#include "exhaustive_gs.h"
%}

%include "logger.h"
%include "global.h"
%include "exhaustive_gs.h"

namespace std {
    %template(DoublePair) pair<double, double>;
    %template(DoublePairVector) vector< pair <double, double> >;
    %template(FloatPair) pair<float, float>;
    %template(FloatPairVector) vector< pair <float, float> >;
    %template(FloatVector) vector<float>;
    %template(IntVector) vector<int>;
    %template(IntVectorVector) vector< vector<int> >;
    %template(StringPair) pair<string, string>;
    %template(StringPairVector) vector< pair<string, string> >;
    %template(StringVector) vector<string>;
    %template(StringVector2D) vector< vector<string> >;
    %template(StringMap) map< string, string >;

    // Iterable container for suggested ground state results returned by SimAnneal
    %template(ConfigVector) vector< pair< vector<int>, float > >;
}

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_2_UNICODE
%}

%extend egs::SimParams {
    
    void egs::SimParams::pySetVExt(std::vector<float> s_vec) {
        egs::FPVec u_vec(s_vec.size());
        for (unsigned int i=0; i<s_vec.size(); i++)
            u_vec[i] = s_vec[i];
        $self->v_ext = u_vec;
    }

    %pythoncode{
        def set_db_locs(self, db_locs):
            if len(db_locs[0]) == 3:
                dbs = IntVectorVector(db_locs)
            elif len(db_locs[0]) == 2:
                dbs = FloatPairVector(db_locs)
            else:
                raise Exception(f'db_locs elements must have length 2 or 3.')
            self.setDBLocs(dbs)

        def set_v_ext(self, v_ext):
            self.pySetVExt(FloatVector(v_ext))

        def set_param(self, pname, pval):
            try:
                self.__swig_setmethods__[pname](self, pval)
            except AttributeError:
                if pname == 'qubo':
                    self.qubo = pval
                elif pname == 'num_threads':
                    self.num_threads = pval
                elif pname == 'base':
                    self.base = pval
                elif pname == 'autofail':
                    self.autofail = pval
                elif pname == 'muzm':
                    self.muzm = pval
                elif pname == 'eps_r':
                    self.eps_r = pval
                elif pname == 'debye':
                    self.debye = pval
                elif pname == 'v_ext':
                    self.set_v_ext(pval)
                else:
                    raise NotImplementedError(f'Provided parameter name {pname} has not been implemented in the EGS python wrapper.')
            except KeyError:
                print(f'set_param error: {pname} is not a valid SimParams parameter.')
                raise

        def print_phys_params(self):
            print(f'muzm: {self.muzm}; eps_r: {self.eps_r}; debye: {self.debye}')
    }
}

%extend egs::EGS {

    // Convert vector of ChargeConfigResult to a vector of pair that has been 
    // specifically defined above as ConfigVector such that SWIG knows how to 
    // generate a suitable container for the results.
    std::vector<std::pair<std::vector<int>, float>> egs::EGS::pyFilteredResults() {
        std::vector<std::pair<std::vector<int>, float>> out_results;
        for (auto result : self->filteredResults()) {
            std::vector<int> conf;
            for (int chg : result.config)
                conf.push_back(chg);
            out_results.push_back(std::make_pair(conf, result.energy));
        }
        return out_results;
    }

    %pythoncode{
        from collections import namedtuple
        ChargeResult = namedtuple('ChargeResult', ['config', 'energy'])

        def gs_results(self):
            configs = []
            for conf in self.pyFilteredResults():
                chg_cfg = [chg for chg in conf[0]]
                configs.append(self.ChargeResult(chg_cfg, conf[1]))
            return configs
    }
}
