cmake_minimum_required(VERSION 3.10)

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

if(DEFINED SIQAD_PLUGINS_ROOT)
    set(EGS_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/exhaustive_gs")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(EGS_INSTALL_DIR "release")
else()
    set(EGS_INSTALL_DIR "debug")
ENDIF()

set(EGS_FILES
    LICENSE
    exhaustive_gs.physeng
    src/exhaustive_gs.py
)

install(FILES ${EGS_FILES} DESTINATION ${EGS_INSTALL_DIR})

if(COMMAND symlink_siqadtools AND NOT WIN32)
    symlink_siqadtools(${EGS_INSTALL_DIR})
endif()
